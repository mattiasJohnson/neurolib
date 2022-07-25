import numpy as np
import numba

from . import loadDefaultParams as dp
from ...utils import model_utils as mu


def timeIntegration(params):
    """
    Sets up the parameters for time integration

    Return:
      rates_exc:  n_nodes_ctx*L array   : containing the exc. neuron rates in kHz time series for each aln node
      rates_inh:  n_nodes_ctx*L array   : containing the inh. neuron rates in kHz time series for each aln node
      t:          L array               : time in ms
      mufe:       n_nodes_ctx vector    : final value of mufe for each node
      mufi:       n_nodes_ctx vector    : final value of mufi for each node
      IA:         n_nodes_ctx vector    : final value of IA   for each node
      seem:       n_nodes_ctx vector    : final value of seem for each node
      seim:       n_nodes_ctx vector    : final value of seim for each node
      siem:       n_nodes_ctx vector    : final value of siem for each node
      siim:       n_nodes_ctx vector    : final value of siim for each node
      seev:       n_nodes_ctx vector    : final value of seev for each node
      seiv:       n_nodes_ctx vector    : final value of seiv for each node
      siev:       n_nodes_ctx vector    : final value of siev for each node
      siiv:       n_nodes_ctx vector    : final value of siiv for each node
      # TODO: Add units to thalamic time series like V_t:
      V_t,        n_nodes_thal*L array  : contaning the exc. thalamic membrane potential
      V_r,        n_nodes_thal*L array  : contaning the inh. thalamic membrane potential
      Q_t,        n_nodes_thal*L array  : contaning the exc. thalamic mean fire rates
      Q_r,        n_nodes_thal*L array  : contaning the inh. thalamic mean fire rates
      Ca,         n_nodes_thal vector   : final value for Ca for each node
      h_T_t,      n_nodes_thal vector   : final value for h_T_t for each node
      h_T_r,      n_nodes_thal vector   : final value for h_T_r for each node
      m_h1,       n_nodes_thal vector   : final value for m_h1 for each node
      m_h2,       n_nodes_thal vector   : final value for m_h2 for each node
      s_et,       n_nodes_thal vector   : final value for s_et for each node
      s_gt,       n_nodes_thal vector   : final value for s_gt for each node
      s_er,       n_nodes_thal vector   : final value for s_er for each node
      s_gr,       n_nodes_thal vector   : final value for s_gr for each node
      ds_et,      n_nodes_thal vector   : final value for ds_et for each node
      ds_gt,      n_nodes_thal vector   : final value for ds_gt for each node
      ds_er,      n_nodes_thal vector   : final value for ds_er for each node
      ds_gr,      n_nodes_thal vector   : final value for ds_gr for each node

    :param params: Parameter dictionary of the model
    :type params: dict
    :return: Integrated activity variables of the model
    :rtype: (numpy.ndarray,)
    """

    dt = params["dt"]  # Time step for the Euler intergration (ms)
    sqrt_dt = np.sqrt(dt)
    duration = params["duration"]  # Simulation duration (ms)
    # Floating point issue in np.arange() workaraound: use integers in np.arange()
    t = np.arange(1, round(duration, 6) / dt + 1) * dt  # Time variable (ms)
    RNGseed = params["seed"]  # seed for RNG
    # set to 0 for faster computation
    cortical_noise = params["cortical_noise"]  # TODO: for debug, remove when not needed.

    # ------------------------------------------------------------------------
    # global coupling parameters

    # Connectivity matric
    # Interareal relative coupling strengths (values between 0 and 1), Cmat(i,j) connnection from jth to ith
    params["Cmat_scaled"] = scaleCmat(
        params["Cmat"],
        params["scale_ctx_to_ctx"],
        params["scale_ctx_to_thal"],
        params["scale_thal_to_ctx"],
        params["scale_thal_to_thal"],
        params["n_nodes_ctx"],
        params["n_nodes_thal"]
    )
    Cmat = params["Cmat_scaled"]
    c_gl = params["c_gl"]  # EPSP amplitude between areas
    Ke_gl = params["Ke_gl"]  # number of incoming E connections (to E population) from each area

    # N = len(Cmat)  # Number of areas
    n_nodes_ctx = params["n_nodes_ctx"]  # Number of cortical areas
    n_nodes_thal = params["n_nodes_thal"]  # Number of thalamic areas
    n_nodes_tot = params["n_nodes_tot"]

    # Interareal connection delay
    lengthMat = params["lengthMat"]
    signalV = params["signalV"]

    # TODO: what should delays be?
    Dmat = dp.computeDelayMatrix(
        lengthMat, signalV
    )  # Interareal connection delays, Dmat(i,j) Connnection from jth node to ith (ms)
    np.fill_diagonal(Dmat[:n_nodes_ctx, :n_nodes_ctx], params["de"])  # Cortex self-delays == de
    Dmat_ndt = np.around(Dmat / dt).astype(int)  # delay matrix in multiples of dt

    # ------------------------------------------------------------------------

    # local network (area) parameters [identical for all areas for now]

    ### model parameters
    filter_sigma = params["filter_sigma"]

    # distributed delay between areas, not tested, but should work
    # distributed delay is implemented by a convolution with the delay kernel
    # the convolution is represented as a linear ODE with the timescale that
    # corresponds to the width of the delay distribution
    distr_delay = params["distr_delay"]

    # external input parameters:
    tau_ou = params["tau_ou"]  # Parameter of the Ornstein-Uhlenbeck process for the external input(ms)
    # Parameter of the Ornstein-Uhlenbeck (OU) process for the external input ( mV/ms/sqrt(ms) )
    sigma_ou = params["sigma_ou"]
    mue_ext_mean = params["mue_ext_mean"]  # Mean external excitatory input (OU process) (mV/ms)
    mui_ext_mean = params["mui_ext_mean"]  # Mean external inhibitory input (OU process) (mV/ms)
    sigmae_ext = params["sigmae_ext"]  # External exc input standard deviation ( mV/sqrt(ms) )
    sigmai_ext = params["sigmai_ext"]  # External inh input standard deviation ( mV/sqrt(ms) )

    # recurrent coupling parameters
    Ke = params["Ke"]  # Recurrent Exc coupling. "EE = IE" assumed for act_dep_coupling in current implementation
    Ki = params["Ki"]  # Recurrent Exc coupling. "EI = II" assumed for act_dep_coupling in current implementation

    # Recurrent connection delays
    de = params["de"]  # Local constant delay "EE = IE" (ms)
    di = params["di"]  # Local constant delay "EI = II" (ms)

    tau_se = params["tau_se"]  # Synaptic decay time constant for exc. connections "EE = IE" (ms)
    tau_si = params["tau_si"]  # Synaptic decay time constant for inh. connections  "EI = II" (ms)
    tau_de = params["tau_de"]
    tau_di = params["tau_di"]

    cee = params["cee"]  # strength of exc. connection
    #  -> determines ePSP magnitude in state-dependent way (in the original model)
    cie = params["cie"]  # strength of inh. connection
    #   -> determines iPSP magnitude in state-dependent way (in the original model)
    cei = params["cei"]
    cii = params["cii"]

    # Recurrent connections coupling strength
    Jee_max = params["Jee_max"]  # ( mV/ms )
    Jei_max = params["Jei_max"]  # ( mV/ms )
    Jie_max = params["Jie_max"]  # ( mV/ms )
    Jii_max = params["Jii_max"]  # ( mV/ms )

    # rescales c's here: multiplication with tau_se makes
    # the increase of s subject to a single input spike invariant to tau_se
    # division by J ensures that mu = J*s will result in a PSP of exactly c
    # for a single spike!

    cee = cee * tau_se / Jee_max  # ms
    cie = cie * tau_se / Jie_max  # ms
    cei = cei * tau_si / abs(Jei_max)  # ms
    cii = cii * tau_si / abs(Jii_max)  # ms
    c_gl = c_gl * tau_se / Jee_max  # ms

    # neuron model parameters
    a = params["a"]  # Adaptation coupling term ( nS )
    b = params["b"]  # Spike triggered adaptation ( pA )
    EA = params["EA"]  # Adaptation reversal potential ( mV )
    tauA = params["tauA"]  # Adaptation time constant ( ms )
    # if params below are changed, preprocessing required
    C = params["C"]  # membrane capacitance ( pF )
    gL = params["gL"]  # Membrane conductance ( nS )
    EL = params["EL"]  # Leak reversal potential ( mV )
    DeltaT = params["DeltaT"]  # Slope factor ( EIF neuron ) ( mV )
    VT = params["VT"]  # Effective threshold (in exp term of the aEIF model)(mV)
    Vr = params["Vr"]  # Membrane potential reset value (mV)
    Vs = params["Vs"]  # Cutoff or spike voltage value, determines the time of spike (mV)
    Tref = params["Tref"]  # Refractory time (ms)
    taum = C / gL  # membrane time constant

    # ------------------------------------------------------------------------

    # Lookup tables for the transfer functions
    precalc_r, precalc_V, precalc_tau_mu, precalc_tau_sigma = (
        params["precalc_r"],
        params["precalc_V"],
        params["precalc_tau_mu"],
        params["precalc_tau_sigma"],
    )

    # parameter for the lookup tables
    dI = params["dI"]
    ds = params["ds"]
    sigmarange = params["sigmarange"]
    Irange = params["Irange"]

    # Initialization
    ndt_de = np.around(de / dt).astype(int)
    ndt_di = np.around(di / dt).astype(int)

    rd_exc = np.zeros((n_nodes_tot, n_nodes_tot))  # kHz  rd_exc(i,j): Connection from jth node to ith
    rd_inh = np.zeros(n_nodes_ctx)

    max_global_delay = max(np.max(Dmat_ndt), ndt_de, ndt_di)
    startind = int(max_global_delay + 1)

    # state variable arrays, have length of t + startind
    # they store initial conditions AND simulated data
    rates_exc = np.zeros((n_nodes_ctx, startind + len(t)))
    rates_inh = np.zeros((n_nodes_ctx, startind + len(t)))
    IA = np.zeros((n_nodes_ctx, startind + len(t)))

    # ------------------------------------------------------------------------
    # Set initial values
    mufe = params["mufe_init"].copy()  # Filtered mean input (mu) for exc. population
    mufi = params["mufi_init"].copy()  # Filtered mean input (mu) for inh. population
    IA_init = params["IA_init"].copy()  # Adaptation current (pA)
    seem = params["seem_init"].copy()  # Mean exc synaptic input
    seim = params["seim_init"].copy()
    seev = params["seev_init"].copy()  # Exc synaptic input variance
    seiv = params["seiv_init"].copy()
    siim = params["siim_init"].copy()  # Mean inh synaptic input
    siem = params["siem_init"].copy()
    siiv = params["siiv_init"].copy()  # Inh synaptic input variance
    siev = params["siev_init"].copy()

    mue_ou = params["mue_ou"].copy()  # Mean of external exc OU input (mV/ms)
    mui_ou = params["mui_ou"].copy()  # Mean of external inh ON inout (mV/ms)

    # Set the initial firing rates.
    # if initial values are just a Nx1 array
    if np.shape(params["rates_exc_init"])[1] == 1:
        # repeat the 1-dim value startind times
        rates_exc_init = params["rates_exc_init"] * np.ones((n_nodes_ctx, startind))  # kHz
        rates_inh_init = params["rates_inh_init"] * np.ones((n_nodes_ctx, startind))  # kHz
        # set initial adaptation current
        IA_init = params["IA_init"] * np.ones((n_nodes_ctx, startind))
    # if initial values are a Nxt array
    else:
        rates_exc_init = params["rates_exc_init"][:, -startind:]
        rates_inh_init = params["rates_inh_init"][:, -startind:]
        IA_init = params["IA_init"][:, -startind:]

    np.random.seed(RNGseed)

    # Save the noise in the rates array to save memory
    if cortical_noise:
        rates_exc[:, startind:] = np.random.standard_normal((n_nodes_ctx, len(t)))
        rates_inh[:, startind:] = np.random.standard_normal((n_nodes_ctx, len(t)))
    else:
        rates_exc[:, startind:] = np.zeros((n_nodes_ctx, len(t)))
        rates_inh[:, startind:] = np.zeros((n_nodes_ctx, len(t)))

    # Set the initial conditions
    rates_exc[:, :startind] = rates_exc_init
    rates_inh[:, :startind] = rates_inh_init
    IA[:, :startind] = IA_init

    # TODO: why not just have a single variable instead of array if all the noise is saved in rates_exc anyway?
    # Guess: maybe it won't make a difference for performance (in which case the local variable should be easier)
    # Or in optimisation maybe having n_nodes_ctx local variables in each loop performancewise is the same as an array
    noise_exc = np.zeros((n_nodes_ctx,))
    noise_inh = np.zeros((n_nodes_ctx,))

    # tile external inputs to appropriate shape
    ext_exc_current = mu.adjustArrayShape(params["ext_exc_current"], rates_exc)
    ext_inh_current = mu.adjustArrayShape(params["ext_inh_current"], rates_exc)
    ext_exc_rate = mu.adjustArrayShape(params["ext_exc_rate"], rates_exc)
    ext_inh_rate = mu.adjustArrayShape(params["ext_inh_rate"], rates_exc)

    # ------------------------------------------------------------------------
    # Thalamus parameters
    # ------------------------------------------------------------------------

    tau = params["tau"]
    Q_max = params["Q_max"]
    C1 = params["C1"]
    theta = params["theta"]
    sigma = params["sigma"]
    g_L = params["g_L"]
    E_L = params["E_L"]
    g_AMPA_t = params["g_AMPA_t"]
    g_AMPA_r = params["g_AMPA_r"]
    g_GABA_t = params["g_GABA_t"]
    g_GABA_r = params["g_GABA_r"]
    E_AMPA = params["E_AMPA"]
    E_GABA = params["E_GABA"]
    g_LK_t = params["g_LK_t"]
    g_LK_r = params["g_LK_r"]
    E_K = params["E_K"]
    g_T_t = params["g_T_t"]
    g_T_r = params["g_T_r"]
    E_Ca = params["E_Ca"]
    g_h = params["g_h"]
    g_inc = params["g_inc"]
    E_h = params["E_h"]
    C_m = params["C_m"]
    alpha_Ca = params["alpha_Ca"]
    Ca_0 = params["Ca_0"]
    tau_Ca = params["tau_Ca"]
    k1 = params["k1"]
    k2 = params["k2"]
    k3 = params["k3"]
    k4 = params["k4"]
    n_P = params["n_P"]
    gamma_e = params["gamma_e"]
    gamma_r = params["gamma_r"]
    d_phi = params["d_phi"]
    shift_HA = params["shift_HA"]
    N_rt = params["N_rt"]
    N_tr = params["N_tr"]
    N_rr = params["N_rr"]

    ext_current_t = params["ext_current_t"]
    ext_current_r = params["ext_current_r"]

    # model output
    V_t = np.zeros((n_nodes_thal, startind + len(t)))
    V_r = np.zeros((n_nodes_thal, startind + len(t)))
    Q_t = np.zeros((n_nodes_thal, startind + len(t)))
    Q_r = np.zeros((n_nodes_thal, startind + len(t)))
    # Set initial Thalamus membrane potentials
    # if initial values are just a Nx1 array
    if np.shape(params["V_t_init"])[1] == 1:
        # repeat the 1-dim value startind times
        V_t_init = params["V_t_init"] * np.ones((n_nodes_thal, startind))
        V_r_init = params["V_r_init"] * np.ones((n_nodes_thal, startind))
        Q_t_init = params["Q_t_init"] * np.ones((n_nodes_thal, startind))
        Q_r_init = params["Q_r_init"] * np.ones((n_nodes_thal, startind))
    # if initial values are a Nxt array
    else:
        V_t_init = params["V_t_init"][:, -startind:]
        V_r_init = params["V_r_init"][:, -startind:]
        Q_t_init = params["Q_t_init"][:, -startind:]
        Q_r_init = params["Q_r_init"][:, -startind:]
    # init
    V_t[:, :startind] = V_t_init
    V_r[:, :startind] = V_r_init
    Q_t[:, :startind] = Q_t_init
    Q_r[:, :startind] = Q_r_init
    # TODO: why convert first to array in loadDefault and now to float?
    Ca = params["Ca_init"]
    h_T_t = params["h_T_t_init"]
    h_T_r = params["h_T_r_init"]
    m_h1 = params["m_h1_init"]
    m_h2 = params["m_h2_init"]
    s_et = params["s_et_init"]
    s_gt = params["s_gt_init"]
    s_er = params["s_er_init"]
    s_gr = params["s_gr_init"]
    ds_et = params["ds_et_init"]
    ds_gt = params["ds_gt_init"]
    ds_er = params["ds_er_init"]
    ds_gr = params["ds_gr_init"]

    include_thal_rowsums = params["include_thal_rowsums"]

    if include_thal_rowsums:
        thal_rowsums = np.zeros((n_nodes_thal, startind + len(t)))
        I_T_t_array = np.zeros((n_nodes_thal, startind + len(t)))
        I_T_r_array = np.zeros((n_nodes_thal, startind + len(t)))
        I_h_array = np.zeros((n_nodes_thal, startind + len(t)))
        Ca_array = np.zeros((n_nodes_thal, startind + len(t)))
    else:
        thal_rowsums = np.zeros((1,1))  # Dummy variable, numba requires same type even when not used
        I_T_t_array = np.zeros((1,1))
        I_T_r_array = np.zeros((1,1))
        I_h_array = np.zeros((1,1))
        Ca_array = np.zeros((1,1))

    noise_thalamus = np.random.standard_normal(len(t))

    return timeIntegration_njit_elementwise(
        dt,
        duration,
        distr_delay,
        filter_sigma,
        Cmat,
        Dmat,
        c_gl,
        Ke_gl,
        tau_ou,
        sigma_ou,
        mue_ext_mean,
        mui_ext_mean,
        sigmae_ext,
        sigmai_ext,
        Ke,
        Ki,
        de,
        di,
        tau_se,
        tau_si,
        tau_de,
        tau_di,
        cee,
        cie,
        cii,
        cei,
        Jee_max,
        Jei_max,
        Jie_max,
        Jii_max,
        a,
        b,
        EA,
        tauA,
        C,
        gL,
        EL,
        DeltaT,
        VT,
        Vr,
        Vs,
        Tref,
        taum,
        mufe,
        mufi,
        IA,
        seem,
        seim,
        seev,
        seiv,
        siim,
        siem,
        siiv,
        siev,
        precalc_r,
        precalc_V,
        precalc_tau_mu,
        precalc_tau_sigma,
        dI,
        ds,
        sigmarange,
        Irange,
        n_nodes_ctx,
        Dmat_ndt,
        t,
        rates_exc,
        rates_inh,
        rd_exc,
        rd_inh,
        sqrt_dt,
        startind,
        ndt_de,
        ndt_di,
        mue_ou,
        mui_ou,
        ext_exc_rate,
        ext_inh_rate,
        ext_exc_current,
        ext_inh_current,
        noise_exc,
        noise_inh,
        # Thalamus
        # startind,
        # t,
        # dt,
        # sqrt_dt,
        Q_max,
        C1,
        theta,
        sigma,
        g_L,
        E_L,
        g_AMPA_t,
        g_AMPA_r,
        g_GABA_t,
        g_GABA_r,
        E_AMPA,
        E_GABA,
        g_LK_t,
        g_LK_r,
        E_K,
        g_T_t,
        g_T_r,
        E_Ca,
        g_h,
        g_inc,
        E_h,
        C_m,
        tau,
        alpha_Ca,
        Ca_0,
        tau_Ca,
        k1,
        k2,
        k3,
        k4,
        n_P,
        gamma_e,
        gamma_r,
        d_phi,
        shift_HA,
        noise_thalamus,
        ext_current_t,
        ext_current_r,
        N_rt,
        N_tr,
        N_rr,
        V_t,
        V_r,
        Q_t,
        Q_r,
        Ca,
        h_T_t,
        h_T_r,
        m_h1,
        m_h2,
        s_et,
        s_gt,
        s_er,
        s_gr,
        ds_et,
        ds_gt,
        ds_er,
        ds_gr,
        n_nodes_thal,
        cortical_noise,
        thal_rowsums,
        include_thal_rowsums,
        I_T_t_array,
        I_T_r_array,
        I_h_array,
        Ca_array,
    )


@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64, "idx1": numba.int64, "idy1": numba.int64})
def timeIntegration_njit_elementwise(
    dt,
    duration,
    distr_delay,
    filter_sigma,
    Cmat,
    Dmat,
    c_gl,
    Ke_gl,
    tau_ou,
    sigma_ou,
    mue_ext_mean,
    mui_ext_mean,
    sigmae_ext,
    sigmai_ext,
    Ke,
    Ki,
    de,
    di,
    tau_se,
    tau_si,
    tau_de,
    tau_di,
    cee,
    cie,
    cii,
    cei,
    Jee_max,
    Jei_max,
    Jie_max,
    Jii_max,
    a,
    b,
    EA,
    tauA,
    C,
    gL,
    EL,
    DeltaT,
    VT,
    Vr,
    Vs,
    Tref,
    taum,
    mufe,
    mufi,
    IA,
    seem,
    seim,
    seev,
    seiv,
    siim,
    siem,
    siiv,
    siev,
    precalc_r,
    precalc_V,
    precalc_tau_mu,
    precalc_tau_sigma,
    dI,
    ds,
    sigmarange,
    Irange,
    n_nodes_ctx,
    Dmat_ndt,
    t,
    rates_exc,
    rates_inh,
    rd_exc,
    rd_inh,
    sqrt_dt,
    startind,
    ndt_de,
    ndt_di,
    mue_ou,
    mui_ou,
    ext_exc_rate,
    ext_inh_rate,
    ext_exc_current,
    ext_inh_current,
    noise_exc,
    noise_inh,
    # Thalamus
    # startind,
    # t,
    # dt,
    # sqrt_dt,
    Q_max,
    C1,
    theta,
    sigma,
    g_L,
    E_L,
    g_AMPA_t,
    g_AMPA_r,
    g_GABA_t,
    g_GABA_r,
    E_AMPA,
    E_GABA,
    g_LK_t,
    g_LK_r,
    E_K,
    g_T_t,
    g_T_r,
    E_Ca,
    g_h,
    g_inc,
    E_h,
    C_m,
    tau,
    alpha_Ca,
    Ca_0,
    tau_Ca,
    k1,
    k2,
    k3,
    k4,
    n_P,
    gamma_e,
    gamma_r,
    d_phi,
    shift_HA,
    noise_thalamus,
    ext_current_t,
    ext_current_r,
    N_rt,
    N_tr,
    N_rr,
    V_t,
    V_r,
    Q_t,
    Q_r,
    Ca,
    h_T_t,
    h_T_r,
    m_h1,
    m_h2,
    s_et,
    s_gt,
    s_er,
    s_gr,
    ds_et,
    ds_gt,
    ds_er,
    ds_gr,
    n_nodes_thal,
    cortical_noise,
    thal_rowsums,
    include_thal_rowsums,
    I_T_t_array,
    I_T_r_array,
    I_h_array,
    Ca_array,
):

    # Global
    n_nodes_tot = n_nodes_ctx + n_nodes_thal

    # Cortex
    # squared Jee_max
    sq_Jee_max = Jee_max**2
    sq_Jei_max = Jei_max**2
    sq_Jie_max = Jie_max**2
    sq_Jii_max = Jii_max**2

    # initialize so we don't get an error when returning
    rd_exc_rhs = 0.0
    rd_inh_rhs = 0.0
    sigmae_f_rhs = 0.0
    sigmai_f_rhs = 0.0

    if filter_sigma:
        sigmae_f = sigmae_ext
        sigmai_f = sigmai_ext

    # Thalamus functions

    def _firing_rate(voltage):
        return Q_max / (1.0 + np.exp(-C1 * (voltage - theta) / sigma))

    def _leak_current(voltage):
        return g_L * (voltage - E_L)

    def _potassium_leak_current(voltage, g_LK):
        return g_LK * (voltage - E_K)

    def _syn_exc_current(voltage, synaptic_rate, g_AMPA):
        return g_AMPA * synaptic_rate * (voltage - E_AMPA)

    def _syn_inh_current(voltage, synaptic_rate, g_GABA):
        return g_GABA * synaptic_rate * (voltage - E_GABA)

    ### integrate ODE system:
    for i in range(startind, startind + len(t)):

        # -------------------------------------------------------------
        # Cortex
        # -------------------------------------------------------------

        if not distr_delay:
            # Get the input from one node into another from the rates at time t - connection_delay - 1
            # remark: assume Kie == Kee and Kei == Kii
            for to_node in range(n_nodes_tot):
                # Cortical input
                for from_node in range(n_nodes_ctx):
                    # rd_exc(i,j) delayed input rate from population j to population i
                    rd_exc[to_node, from_node] = (
                        rates_exc[from_node, i - Dmat_ndt[to_node, from_node] - 1] * 1e-3
                    )  # convert Hz to kHz

                # Thalamic input
                for idx_thal in range(n_nodes_thal):
                    from_node = idx_thal + n_nodes_ctx
                    rd_exc[to_node, from_node] = (
                        Q_t[idx_thal, i - Dmat_ndt[to_node, from_node] - 1] * 1e-3
                    )  # convert Hz to kHz

            # Cortical inhibitory delayed input
            for idx_ctx in range(n_nodes_ctx):
                # Warning: this is a vector and not a matrix as rd_exc
                rd_inh[idx_ctx] = rates_inh[idx_ctx, i - ndt_di - 1] * 1e-3  # convert Hz to kHz

        # loop through all the cortical nodes
        for no in range(n_nodes_ctx):

            # To save memory, noise is saved in the rates array
            noise_exc[no] = rates_exc[no, i]
            noise_inh[no] = rates_inh[no, i]

            if cortical_noise:
                mue = Jee_max * seem[no] + Jei_max * seim[no] + mue_ou[no] + ext_exc_current[no, i]
                mui = Jie_max * siem[no] + Jii_max * siim[no] + mui_ou[no] + ext_inh_current[no, i]
            else:
                mue = Jee_max * seem[no] + Jei_max * seim[no] + ext_exc_current[no, i]
                mui = Jie_max * siem[no] + Jii_max * siim[no] + ext_inh_current[no, i]

            # compute row sum of Cmat*rd_exc and Cmat**2*rd_exc
            rowsum = 0
            rowsumsq = 0
            for col in range(n_nodes_tot):
                rowsum = rowsum + Cmat[no, col] * rd_exc[no, col]
                rowsumsq = rowsumsq + Cmat[no, col] ** 2 * rd_exc[no, col]

            # z1: weighted sum of delayed rates, weights=c*K
            z1ee = (
                cee * Ke * rd_exc[no, no]  # Self-connection (kHz)
                + c_gl * Ke_gl * rowsum  # Thalamus enters
                + c_gl * Ke_gl * ext_exc_rate[no, i]  # Set to 0 in paper
            )  # rate from other regions + exc_ext_rate
            z1ei = cei * Ki * rd_inh[no]  # kHz
            z1ie = (
                cie * Ke * rd_exc[no, no] + c_gl * Ke_gl * ext_inh_rate[no, i]
            )  # first test of external rate input to inh. population
            z1ii = cii * Ki * rd_inh[no]
            # z2: weighted sum of delayed rates, weights=c^2*K (see thesis last ch.)
            z2ee = (
                cee**2 * Ke * rd_exc[no, no] + c_gl**2 * Ke_gl * rowsumsq + c_gl**2 * Ke_gl * ext_exc_rate[no, i]
            )
            z2ei = cei**2 * Ki * rd_inh[no]
            z2ie = (
                cie**2 * Ke * rd_exc[no, no] + c_gl**2 * Ke_gl * ext_inh_rate[no, i]
            )  # external rate input to inh. population
            z2ii = cii**2 * Ki * rd_inh[no]

            sigmae = np.sqrt(
                2 * sq_Jee_max * seev[no] * tau_se * taum / ((1 + z1ee) * taum + tau_se)
                + 2 * sq_Jei_max * seiv[no] * tau_si * taum / ((1 + z1ei) * taum + tau_si)
                + sigmae_ext**2
            )  # mV/sqrt(ms)

            sigmai = np.sqrt(
                2 * sq_Jie_max * siev[no] * tau_se * taum / ((1 + z1ie) * taum + tau_se)
                + 2 * sq_Jii_max * siiv[no] * tau_si * taum / ((1 + z1ii) * taum + tau_si)
                + sigmai_ext**2
            )  # mV/sqrt(ms)

            if not filter_sigma:
                sigmae_f = sigmae
                sigmai_f = sigmai

            # Read the transfer function from the lookup table
            # -------------------------------------------------------------

            # ------- excitatory population
            # mufe[no] - IA[no] / C is the total current of the excitatory population
            xid1, yid1, dxid, dyid = fast_interp2_opt(
                sigmarange, ds, sigmae_f, Irange, dI, mufe[no] - IA[no, i - 1] / C
            )
            xid1, yid1 = int(xid1), int(yid1)

            rates_exc[no, i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid) * 1e3  # convert kHz to Hz
            Vmean_exc = interpolate_values(precalc_V, xid1, yid1, dxid, dyid)
            tau_exc = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
            if filter_sigma:
                tau_sigmae_eff = interpolate_values(precalc_tau_sigma, xid1, yid1, dxid, dyid)

            # ------- inhibitory population
            #  mufi[no] are the (filtered) currents of the inhibitory population
            xid1, yid1, dxid, dyid = fast_interp2_opt(sigmarange, ds, sigmai_f, Irange, dI, mufi[no])
            xid1, yid1 = int(xid1), int(yid1)

            rates_inh[no, i] = interpolate_values(precalc_r, xid1, yid1, dxid, dyid) * 1e3
            # Vmean_inh = interpolate_values(precalc_V, xid1, yid1, dxid, dyid) # not used
            tau_inh = interpolate_values(precalc_tau_mu, xid1, yid1, dxid, dyid)
            if filter_sigma:
                tau_sigmai_eff = interpolate_values(precalc_tau_sigma, xid1, yid1, dxid, dyid)

            # -------------------------------------------------------------

            # now everything available for r.h.s:

            mufe_rhs = (mue - mufe[no]) / tau_exc
            mufi_rhs = (mui - mufi[no]) / tau_inh

            # rate has to be kHz
            IA_rhs = (a * (Vmean_exc - EA) - IA[no, i - 1] + tauA * b * rates_exc[no, i] * 1e-3) / tauA

            # EQ. 4.43
            if distr_delay:
                rd_exc_rhs = (rates_exc[no, i] * 1e-3 - rd_exc[no, no]) / tau_de
                rd_inh_rhs = (rates_inh[no, i] * 1e-3 - rd_inh[no]) / tau_di

            if filter_sigma:
                sigmae_f_rhs = (sigmae - sigmae_f) / tau_sigmae_eff
                sigmai_f_rhs = (sigmai - sigmai_f) / tau_sigmai_eff

            # integration of synaptic input (eq. 4.36)
            seem_rhs = ((1 - seem[no]) * z1ee - seem[no]) / tau_se
            seim_rhs = ((1 - seim[no]) * z1ei - seim[no]) / tau_si
            siem_rhs = ((1 - siem[no]) * z1ie - siem[no]) / tau_se
            siim_rhs = ((1 - siim[no]) * z1ii - siim[no]) / tau_si
            seev_rhs = ((1 - seem[no]) ** 2 * z2ee + (z2ee - 2 * tau_se * (z1ee + 1)) * seev[no]) / tau_se**2
            seiv_rhs = ((1 - seim[no]) ** 2 * z2ei + (z2ei - 2 * tau_si * (z1ei + 1)) * seiv[no]) / tau_si**2
            siev_rhs = ((1 - siem[no]) ** 2 * z2ie + (z2ie - 2 * tau_se * (z1ie + 1)) * siev[no]) / tau_se**2
            siiv_rhs = ((1 - siim[no]) ** 2 * z2ii + (z2ii - 2 * tau_si * (z1ii + 1)) * siiv[no]) / tau_si**2

            # -------------- integration --------------

            mufe[no] = mufe[no] + dt * mufe_rhs
            mufi[no] = mufi[no] + dt * mufi_rhs
            IA[no, i] = IA[no, i - 1] + dt * IA_rhs

            if distr_delay:
                rd_exc[no, no] = rd_exc[no, no] + dt * rd_exc_rhs
                rd_inh[no] = rd_inh[no] + dt * rd_inh_rhs

            if filter_sigma:
                sigmae_f = sigmae_f + dt * sigmae_f_rhs
                sigmai_f = sigmai_f + dt * sigmai_f_rhs

            seem[no] = seem[no] + dt * seem_rhs
            seim[no] = seim[no] + dt * seim_rhs
            siem[no] = siem[no] + dt * siem_rhs
            siim[no] = siim[no] + dt * siim_rhs
            seev[no] = seev[no] + dt * seev_rhs
            seiv[no] = seiv[no] + dt * seiv_rhs
            siev[no] = siev[no] + dt * siev_rhs
            siiv[no] = siiv[no] + dt * siiv_rhs

            # Ensure the variance does not get negative for low activity
            if seev[no] < 0:
                seev[no] = 0.0

            if siev[no] < 0:
                siev[no] = 0.0

            if seiv[no] < 0:
                seiv[no] = 0.0

            if siiv[no] < 0:
                siiv[no] = 0.0

            # ornstein-uhlenbeck process
            mue_ou[no] = (
                mue_ou[no] + (mue_ext_mean - mue_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_exc[no]
            )  # mV/ms
            mui_ou[no] = (
                mui_ou[no] + (mui_ext_mean - mui_ou[no]) * dt / tau_ou + sigma_ou * sqrt_dt * noise_inh[no]
            )  # mV/ms

        # -------------------------------------------------------------
        # Thalamus
        # -------------------------------------------------------------

        # loop through all the thalamic nodes
        for no in range(n_nodes_thal):

            # leak current
            I_leak_t = _leak_current(V_t[no, i - 1])
            I_leak_r = _leak_current(V_r[no, i - 1])

            # synaptic currents
            I_et = _syn_exc_current(V_t[no, i - 1], s_et[no], g_AMPA_t)
            I_gt = _syn_inh_current(V_t[no, i - 1], s_gt[no], g_GABA_t)
            I_er = _syn_exc_current(V_r[no, i - 1], s_er[no], g_AMPA_r)
            I_gr = _syn_inh_current(V_r[no, i - 1], s_gr[no], g_GABA_r)

            # potassium leak current
            I_LK_t = _potassium_leak_current(V_t[no, i - 1], g_LK_t)
            I_LK_r = _potassium_leak_current(V_r[no, i - 1], g_LK_r)

            # T-type Ca current
            m_inf_T_t = 1.0 / (1.0 + np.exp(-(V_t[no, i - 1] + 59.0) / 6.2))
            m_inf_T_r = 1.0 / (1.0 + np.exp(-(V_r[no, i - 1] + 52.0) / 7.4))
            I_T_t = g_T_t * m_inf_T_t * m_inf_T_t * h_T_t[no] * (V_t[no, i - 1] - E_Ca)
            I_T_r = g_T_r * m_inf_T_r * m_inf_T_r * h_T_r[no] * (V_r[no, i - 1] - E_Ca)

            # h-type current
            I_h = g_h * (m_h1[no] + g_inc * m_h2[no]) * (V_t[no, i - 1] - E_h)

            ### define derivatives
            # membrane potential
            d_V_t = -(I_leak_t + I_et + I_gt + ext_current_t) / tau - (1.0 / C_m) * (I_LK_t + I_T_t + I_h)
            d_V_r = -(I_leak_r + I_er + I_gr + ext_current_r) / tau - (1.0 / C_m) * (I_LK_r + I_T_r)
            # Calcium concentration
            d_Ca = alpha_Ca * I_T_t - (Ca[no] - Ca_0) / tau_Ca
            # channel dynamics
            h_inf_T_t = 1.0 / (1.0 + np.exp((V_t[no, i - 1] + 81.0) / 4.0))
            h_inf_T_r = 1.0 / (1.0 + np.exp((V_r[no, i - 1] + 80.0) / 5.0))
            tau_h_T_t = (
                30.8 + (211.4 + np.exp((V_t[no, i - 1] + 115.2) / 5.0)) / (1.0 + np.exp((V_t[no, i - 1] + 86.0) / 3.2))
            ) / 3.7371928
            tau_h_T_r = (
                85.0 + 1.0 / (np.exp((V_r[no, i - 1] + 48.0) / 4.0) + np.exp(-(V_r[no, i - 1] + 407.0) / 50.0))
            ) / 3.7371928
            d_h_T_t = (h_inf_T_t - h_T_t[no]) / tau_h_T_t
            d_h_T_r = (h_inf_T_r - h_T_r[no]) / tau_h_T_r
            m_inf_h = 1.0 / (1.0 + np.exp((V_t[no, i - 1] + 75.0 + shift_HA) / 5.5))
            tau_m_h = 20.0 + 1000.0 / (np.exp((V_t[no, i - 1] + 71.5) / 14.2) + np.exp(-(V_t[no, i - 1] + 89.0) / 11.6))
            # Calcium channel dynamics
            P_h = k1 * Ca[no]**n_P / (k1 * Ca[no]**n_P + k2)
            d_m_h1 = (m_inf_h * (1.0 - m_h2[no]) - m_h1[no]) / tau_m_h - k3 * P_h * m_h1[no] + k4 * m_h2[no]
            d_m_h2 = k3 * P_h * m_h1[no] - k4 * m_h2[no]
            # synaptic dynamics
            d_s_et = ds_et[no]
            d_s_er = ds_er[no]
            d_s_gt = ds_gt[no]
            d_s_gr = ds_gr[no]

            cortical_rowsum = 0
            for col in range(n_nodes_tot):
                cortical_rowsum = cortical_rowsum + Cmat[no + n_nodes_ctx, col] * rd_exc[no + n_nodes_ctx, col]
                
            # d_ds_et = 0.0
            d_ds_et = gamma_e**2 * (cortical_rowsum - s_et[no]) - 2 * gamma_e * ds_et[no]  # 0 if rowsum == 0 since ds_et[no] == 0
            # d_ds_er = gamma_e**2 * (N_rt * _firing_rate(V_t[no, i - 1]) - s_er[no]) - 2 * gamma_e * ds_er[no]
            d_ds_er = (
                gamma_e**2 * (N_rt * _firing_rate(V_t[no, i - 1]) + cortical_rowsum - s_er[no]) - 2 * gamma_e * ds_er[no]
            )
            d_ds_gt = gamma_r**2 * (N_tr * _firing_rate(V_r[no, i - 1]) - s_gt[no]) - 2 * gamma_r * ds_gt[no]
            d_ds_gr = gamma_r**2 * (N_rr * _firing_rate(V_r[no, i - 1]) - s_gr[no]) - 2 * gamma_r * ds_gr[no]

            ### Euler integration
            V_t[no, i] = V_t[no, i - 1] + dt * d_V_t
            V_r[no, i] = V_r[no, i - 1] + dt * d_V_r
            Q_t[no, i] = _firing_rate(V_t[no, i]) * 1e3  # convert kHz to Hz
            Q_r[no, i] = _firing_rate(V_r[no, i]) * 1e3  # convert kHz to Hz
            Ca[no] = Ca[no] + dt * d_Ca
            h_T_t[no] = h_T_t[no] + dt * d_h_T_t
            h_T_r[no] = h_T_r[no] + dt * d_h_T_r
            m_h1[no] = m_h1[no] + dt * d_m_h1
            m_h2[no] = m_h2[no] + dt * d_m_h2
            s_et[no] = s_et[no] + dt * d_s_et
            s_gt[no] = s_gt[no] + dt * d_s_gt
            s_er[no] = s_er[no] + dt * d_s_er
            s_gr[no] = s_gr[no] + dt * d_s_gr
            # noisy variable
            ds_et[no] = ds_et[no] + dt * d_ds_et + gamma_e**2 * d_phi * sqrt_dt * noise_thalamus[i-startind]
            ds_gt[no] = ds_gt[no] + dt * d_ds_gt
            ds_er[no] = ds_er[no] + dt * d_ds_er
            ds_gr[no] = ds_gr[no] + dt * d_ds_gr

            # Thalamus debug variables
            if include_thal_rowsums:
                thal_rowsums[no, i] = cortical_rowsum
                I_T_t_array[no, i] = I_T_t
                I_T_r_array[no, i] = I_T_r
                I_h_array[no, i] = I_h
                Ca_array[no, i] = Ca[no]

    return (
        t,
        # ALN
        rates_exc,
        rates_inh,
        mufe,
        mufi,
        IA,
        seem,
        seim,
        siem,
        siim,
        seev,
        seiv,
        siev,
        siiv,
        mue_ou,
        mui_ou,
        # Thalamus
        V_t,
        V_r,
        Q_t,
        Q_r,
        Ca,
        h_T_t,
        h_T_r,
        m_h1,
        m_h2,
        s_et,
        s_gt,
        s_er,
        s_gr,
        ds_et,
        ds_gt,
        ds_er,
        ds_gr,
        # Thalamus debug
        thal_rowsums,
        I_T_t_array,
        I_T_r_array,
        I_h_array,
        Ca_array,
    )


@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64})
def interpolate_values(table, xid1, yid1, dxid, dyid):
    output = (
        table[yid1, xid1] * (1 - dxid) * (1 - dyid)
        + table[yid1, xid1 + 1] * dxid * (1 - dyid)
        + table[yid1 + 1, xid1] * (1 - dxid) * dyid
        + table[yid1 + 1, xid1 + 1] * dxid * dyid
    )
    return output


@numba.njit(locals={"idxX": numba.int64, "idxY": numba.int64})
def lookup_no_interp(x, dx, xi, y, dy, yi):

    """
    Return the indices for the closest values for a look-up table
    Choose the closest point in the grid

    x     ... range of x values
    xi    ... interpolation value on x-axis
    dx    ... grid width of x ( dx = x[1]-x[0])
               (same for y)

    return:   idxX and idxY
    """

    if xi > x[0] and xi < x[-1]:
        xid = (xi - x[0]) / dx
        xid_floor = np.floor(xid)
        if xid - xid_floor < dx / 2:
            idxX = xid_floor
        else:
            idxX = xid_floor + 1
    elif xi < x[0]:
        idxX = 0
    else:
        idxX = len(x) - 1

    if yi > y[0] and yi < y[-1]:
        yid = (yi - y[0]) / dy
        yid_floor = np.floor(yid)
        if yid - yid_floor < dy / 2:
            idxY = yid_floor
        else:
            idxY = yid_floor + 1

    elif yi < y[0]:
        idxY = 0
    else:
        idxY = len(y) - 1

    return idxX, idxY


@numba.njit(locals={"xid1": numba.int64, "yid1": numba.int64, "dxid": numba.float64, "dyid": numba.float64})
def fast_interp2_opt(x, dx, xi, y, dy, yi):

    """
    Returns the values needed for interpolation:
    - bilinear (2D) interpolation within ranges,
    - linear (1D) if "one edge" is crossed,
    - corner value if "two edges" are crossed

    x     ... range of the x value
    xi    ... interpolation value on x-axis
    dx    ... grid width of x ( dx = x[1]-x[0] )
    (same for y)

    return:   xid1    ... index of the lower interpolation value
              dxid    ... distance of xi to the lower interpolation value
              (same for y)
    """

    # within all boundaries
    if xi >= x[0] and xi < x[-1] and yi >= y[0] and yi < y[-1]:
        xid = (xi - x[0]) / dx
        xid1 = np.floor(xid)
        dxid = xid - xid1
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1
        return xid1, yid1, dxid, dyid

    # outside one boundary
    if yi < y[0]:
        yid1 = 0
        dyid = 0.0
        if xi >= x[0] and xi < x[-1]:
            xid = (xi - x[0]) / dx
            xid1 = np.floor(xid)
            dxid = xid - xid1

        elif xi < x[0]:
            xid1 = 0
            dxid = 0.0
        else:  # xi >= x(end)
            xid1 = -1
            dxid = 0.0
        return xid1, yid1, dxid, dyid

    if yi >= y[-1]:
        yid1 = -1
        dyid = 0.0
        if xi >= x[0] and xi < x[-1]:
            xid = (xi - x[0]) / dx
            xid1 = np.floor(xid)
            dxid = xid - xid1

        elif xi < x[0]:
            xid1 = 0
            dxid = 0.0

        else:  # xi >= x(end)
            xid1 = -1
            dxid = 0.0
        return xid1, yid1, dxid, dyid

    if xi < x[0]:
        xid1 = 0
        dxid = 0.0
        # We know that yi is within the boundaries
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1
        return xid1, yid1, dxid, dyid

    if xi >= x[-1]:
        xid1 = -1
        dxid = 0.0
        # We know that yi is within the boundaries
        yid = (yi - y[0]) / dy
        yid1 = np.floor(yid)
        dyid = yid - yid1

    return xid1, yid1, dxid, dyid


def scaleCmat(  
    Cmat: np.ndarray,
    scale_ctx_to_ctx: float, 
    scale_ctx_to_thal: float,
    scale_thal_to_ctx: float,
    scale_thal_to_thal: float,
    n_nodes_ctx: int,
    n_nodes_thal: int,
    ) -> np.ndarray:
    """
    Scales the different types of connections in Cmat.
    E.g. ctx_to_thal scales the connections from cortex to thalamus. 
    """

    Cmat_new = Cmat.copy()

    Cmat_new[0:n_nodes_ctx, 0:n_nodes_ctx] *= scale_ctx_to_ctx
    Cmat_new[n_nodes_ctx:n_nodes_ctx+n_nodes_thal, 0:n_nodes_ctx] *= scale_ctx_to_thal
    Cmat_new[0:n_nodes_ctx, n_nodes_ctx:n_nodes_ctx+n_nodes_thal] *= scale_thal_to_ctx
    Cmat_new[n_nodes_ctx:n_nodes_ctx+n_nodes_thal, n_nodes_ctx:n_nodes_ctx+n_nodes_thal] *= scale_thal_to_thal

    return Cmat_new