import os
import numpy as np
import h5py

from ...utils.collections import star_dotdict


def loadDefaultParams(Cmat=None, Dmat=None, lengthMat=None, lookupTableFileName=None, seed=None, n_nodes_ctx=None, n_nodes_thal=None):
    """
    Load default parameters for a network of connected ALN and thalamic nodes.

    :param Cmat: Structural connectivity matrix (adjacency matrix) of coupling strengths. Assumes first `n_nodes_ctx` indices are cortical and following `n_nodes_thal` indices thalamic. If not given, then a model with a single aln and thalamic node will be assumed, defaults to None
    :type Cmat: numpy.ndarray, optional
    :param Dmat: Delay matrix, defaults to None
    :type Dmat: numpy.ndarray, optional
    :param lengthMat: Fiber length matrix, if given then this matrix together with the signal transmission speed parameter `signalV` will be used for computing (and overwrite) the delay matrix Dmat, defaults to None
    :type Dmat: numpy.ndarray, optional
    :param lookUpTableFileName: Filename of lookup table with aln non-linear transfer functions and other precomputed quantities., defaults to aln-precalc/quantities_cascade.h
    :type lookUpTableFileName: str, optional
    :param seed: Seed for the random number generator, defaults to None
    :type seed: int, optional

    :return: A dictionary with the default parameters of the model
    :rtype: dict
    """

    params = star_dotdict({})

    params.model = "thalamocortical"
    params.name = "thalamocortical"
    params.description = "Thalamocortical model with aln and thalamus nodes."

    # runtime parameters
    # thalamus is really sensitive, so either you integrate with very small dt or use an adaptive integration step
    params.dt = 0.01  # ms
    params.duration = 10000  # simulation duration (ms)
    np.random.seed(seed)  # seed for RNG of noise and ICs
    params.seed = seed  # seed needs to be given to constructor `ThalamocorticalModel(seed=seed)` and not `.params["seed"]` for seed to affect ICs.
    params.cortical_noise = True  # TODO: for testing, remove when not needed.

    # options
    params.warn = 0  # warn if limits of lookup tables are exceeded
    params.dosc_version = 0  # if 0, use exponential fit to linear response function
    params.distr_delay = 0  # if 1, use distributed delays instead of fixed
    params.filter_sigma = 0  # if 1, filter sigmae/sigmai
    params.fast_interp = 1  # if 1, Interpolate the value from the look-up table instead of taking the closest value

    # ------------------------------------------------------------------------
    # global whole-brain network parameters
    # ------------------------------------------------------------------------

    if Cmat is None:
        params.Cmat = np.array([[0, 0.12], [1.2, 0]])
        Dmat = np.zeros((2, 2))
        Dmat[0, 1] = 13  # 13 ms delay, thal -> ctx
        Dmat[1, 0] = 13  # 13 ms delay, ctx -> thal
        params.Dmat = Dmat
    else:
        params.Cmat = Cmat.copy()  # coupling matrix
        np.fill_diagonal(params.Cmat, 0)  # no self connections
        params.Dmat = Dmat  # delay matrix
        
    params.lengthMat = lengthMat

    # Number of cortical and thalamic nodes
    if n_nodes_ctx is None:
        n_nodes_ctx = 1

    if n_nodes_thal is None:
        n_nodes_thal = 1

    params.n_nodes_ctx = n_nodes_ctx
    params.n_nodes_thal = n_nodes_thal
    params.n_nodes_tot = params.n_nodes_ctx + params.n_nodes_thal
    
    # Scaling parameters of connectivity
    params.scale_ctx_to_ctx = 1.0
    params.scale_ctx_to_thal = 1.0
    params.scale_thal_to_ctx = 1.0
    params.scale_thal_to_thal = 1.0
    params.Cmat_scaled = params.Cmat.copy()
    
    # Signal transmission speed in mm/ms
    params.signalV = 20.0

    # PSP current amplitude in (mV/ms) (or nA/[C]) for global coupling
    # connections between areas
    params.c_gl = 0.4
    # number of incoming E connections (to E population) from each area
    params.Ke_gl = 250.0

    # ------------------------------------------------------------------------
    # ALN: local E-I node parameters
    # ------------------------------------------------------------------------

    # external input parameters:
    params.tau_ou = 5.0  # ms timescale of ornstein-uhlenbeck (OU) noise
    params.sigma_ou = 0.05  # mV/ms/sqrt(ms) intensity of OU oise
    params.mue_ext_mean = 3.3  # mV/ms mean external input current to E (used to be 3.05 before S3)
    params.mui_ext_mean = 2.0  # mV/ms mean external input current to I

    # Ornstein-Uhlenbeck noise state variables, set to mean input
    # mue_ou will fluctuate around mue_ext_mean (mean of the OU process)
    params.mue_ou = params.mue_ext_mean * np.ones((params.n_nodes_ctx,))
    params.mui_ou = params.mui_ext_mean * np.ones((params.n_nodes_ctx,))

    # external neuronal firing rate input
    params.ext_exc_rate = 0.0  # kHz external excitatory rate drive
    params.ext_inh_rate = 0.0  # kHz external inhibiroty rate drive

    # externaln input currents, same as mue_ext_mean but can be time-dependent!
    params.ext_exc_current = 0.0  # external excitatory input current [mV/ms], C*[]V/s=[]nA
    params.ext_inh_current = 0.0  # external inhibiroty input current [mV/ms]

    # Fokker Planck noise (for N->inf)
    params.sigmae_ext = 1.5  # mV/sqrt(ms) (fixed, for now) [1-5] (Internal noise due to random coupling)
    params.sigmai_ext = 1.5  # mV/sqrt(ms) (fixed, for now) [1-5]

    # recurrent coupling parameters
    params.Ke = 800.0  # Number of excitatory inputs per neuron
    params.Ki = 200.0  # Number of inhibitory inputs per neuron

    # synaptic delays
    params.de = 4.0  # ms local constant delay "EE = IE"
    params.di = 2.0  # ms local constant delay "EI = II"

    # synaptic time constants
    params.tau_se = 2.0  # ms  "EE = IE", for fixed delays
    params.tau_si = 5.0  # ms  "EI = II"

    # time constant for distributed delays (untested)
    params.tau_de = 1.0  # ms  "EE = IE"
    params.tau_di = 1.0  # ms  "EI = II"

    #TODO: these should probably be 0.3 0.5 like aln model
    # PSC amplitudes
    params.cee = 0.3  # mV/ms
    params.cie = 0.3  # AMPA
    params.cei = 0.5  # GABA BrunelWang2003
    params.cii = 0.5

    # Coupling strengths used in Cakan2020
    params.Jee_max = 2.43  # mV/ms
    params.Jie_max = 2.60  # mV/ms
    params.Jei_max = -3.3  # mV/ms [0-(-10)]
    params.Jii_max = -1.64  # mV/ms

    # neuron model parameters
    params.a = 0.0  # nS, can be 15.0
    params.b = 15.0  # pA, can be 40.0
    params.EA = -80.0  # mV
    params.tauA = 1000.0  # ms

    # single neuron paramters - if these are changed, new transfer functions must be precomputed!
    params.C = 200.0  # pF
    params.gL = 10.0  # nS
    params.EL = -65.0  # mV
    params.DeltaT = 1.5  # mV
    params.VT = -50.0  # mV
    params.Vr = -70.0  # mV
    params.Vs = -40.0  # mV
    params.Tref = 1.5  # ms

    # ------------------------------------------------------------------------

    # Generate and set random initial conditions
    (
        # ALN initial conditions:
        params.mufe_init,  # (linear) filtered mean input
        params.mufi_init,
        params.IA_init,  # adaptation current
        params.seem_init,  # mean of fraction of active synapses [0-1] (post-synaptic variable), chap. 4.2
        params.seim_init,  # variance of fraction of active synapses [0-1]
        params.seev_init,
        params.seiv_init,
        params.siim_init,
        params.siem_init,
        params.siiv_init,
        params.siev_init,
        params.rates_exc_init,
        params.rates_inh_init,
    ) = generateRandomICsALN(params.n_nodes_ctx, seed)

    (
        params.voltage_tcr_init,
        params.voltage_trn_init,
        params.rates_tcr_init,
        params.rates_trn_init,
        params.Ca_init,
        params.h_T_t_init,
        params.h_T_r_init,
        params.m_h1_init,
        params.m_h2_init,
        params.s_et_init,
        params.s_gt_init,
        params.s_er_init,
        params.s_gr_init,
        params.ds_et_init,
        params.ds_gt_init,
        params.ds_er_init,
        params.ds_gr_init,
    ) = generateRandomICsThalamus(params.n_nodes_thal, seed)

    # load precomputed aLN transfer functions from hdfs
    if lookupTableFileName is None:
        lookupTableFileName = os.path.join(os.path.dirname(__file__), "aln-precalc", "quantities_cascade.h5")

    hf = h5py.File(lookupTableFileName, "r")
    params.Irange = hf.get("mu_vals")[()]
    params.sigmarange = hf.get("sigma_vals")[()]
    params.dI = params.Irange[1] - params.Irange[0]
    params.ds = params.sigmarange[1] - params.sigmarange[0]

    params.precalc_r = hf.get("r_ss")[()][()]
    params.precalc_V = hf.get("V_mean_ss")[()]
    params.precalc_tau_mu = hf.get("tau_mu_exp")[()]
    params.precalc_tau_sigma = hf.get("tau_sigma_exp")[()]

    # ------------------------------------------------------------------------
    # Default thalamic parameters
    # ------------------------------------------------------------------------

    # local parameters for both populations
    params.tau = 20.0
    params.Q_max = 400.0e-3  # 1/ms
    params.theta = -58.5  # mV
    params.sigma = 6.0
    params.C1 = 1.8137993642
    params.C_m = 1.0  # muF/cm^2
    params.gamma_e = 70.0e-3  # 1/ms
    params.gamma_r = 100.0e-3  # 1/ms
    params.g_L = 1.0  # AU
    params.g_GABA_t = 1.0  # ms
    params.g_GABA_r = 1.0  # ms
    params.g_AMPA_t = 1.0  # ms
    params.g_AMPA_r = 1.0  # ms
    # params.g_LK = 0.018  # mS/cm^2
    params.E_AMPA = 0.0  # mV
    params.E_GABA = -70.0  # mV
    params.E_L = -70.0  # mV
    params.E_K = -100.0  # mV
    params.E_Ca = 120.0  # mV

    # specific thalamo-cortical neurons population - TCR (excitatory)
    params.g_T_t = 3.0  # mS/cm^2
    params.g_LK_t = 0.032  # mS/cm^2
    params.g_h = 0.062  # mS/cm^2
    params.E_h = -40.0  # mV
    params.alpha_Ca = -51.8e-6  # nmol
    params.tau_Ca = 10.0  # ms
    params.Ca_0 = 2.4e-4
    params.k1 = 2.5e7
    params.k2 = 4.0e-4
    params.k3 = 1.0e-1
    params.k4 = 1.0e-3
    params.n_P = 4.0
    params.g_inc = 2.0
    # connectivity
    params.N_tr = 5.0
    # noise
    params.d_phi = 0.0
    params.shift_HA = 0.0 # mV

    # specific thalamic reticular nuclei population - TRN (inhibitory)
    params.g_T_r = 2.3  # mS/cm^2
    params.g_LK_r = 0.032  # mS/cm^2
    # connectivity
    params.N_rt = 3.0
    params.N_rr = 25.0

    # external input
    params.ext_current_t = 0.0
    params.ext_current_r = 0.0

    return params


def computeDelayMatrix(lengthMat, signalV, segmentLength=1):
    """
    Compute the delay matrix from the fiber length matrix and the signal
    velocity

        :param lengthMat:       A matrix containing the connection length in
            segment
        :param signalV:         Signal velocity in m/s
        :param segmentLength:   Length of a single segment in mm

        :returns:    A matrix of connexion delay in ms
    """

    normalizedLenMat = lengthMat * segmentLength
    if signalV > 0:
        Dmat = normalizedLenMat / signalV  # Interareal delays in ms
    else:
        Dmat = lengthMat * 0.0
    return Dmat


def generateRandomICsALN(n_nodes_ctx, seed=None):
    """
    Generates random initial conditions for the aln part of the interareal network

    :params n_nodes_ctx: number of cortical nodes.

    :returns:   A tuple of length 9 representing initial state of the aln part of the model.
                All the elements are `n_nodes_ctx` long numpy arrays representing:
                mufe_init, IA_init, mufi_init, sem_init, sev_init,
                sim_init, siv_init, rates_exc_init, rates_inh_init
    """
    np.random.seed(seed)

    # TODO: Why are all not same shape? Why rates_exc_init two dimensional but most other ones one-dimensional?
    mufe_init = 3 * np.random.uniform(0, 1, (n_nodes_ctx,))  # mV/ms
    mufi_init = 3 * np.random.uniform(0, 1, (n_nodes_ctx,))  # mV/ms
    seem_init = 0.5 * np.random.uniform(0, 1, (n_nodes_ctx,))
    seim_init = 0.5 * np.random.uniform(0, 1, (n_nodes_ctx,))
    seev_init = 0.001 * np.random.uniform(0, 1, (n_nodes_ctx,))
    seiv_init = 0.001 * np.random.uniform(0, 1, (n_nodes_ctx,))
    siim_init = 0.5 * np.random.uniform(0, 1, (n_nodes_ctx,))
    siem_init = 0.5 * np.random.uniform(0, 1, (n_nodes_ctx,))
    siiv_init = 0.01 * np.random.uniform(0, 1, (n_nodes_ctx,))
    siev_init = 0.01 * np.random.uniform(0, 1, (n_nodes_ctx,))
    rates_exc_init = 0.01 * np.random.uniform(0, 1, (n_nodes_ctx, 1))
    rates_inh_init = 0.01 * np.random.uniform(0, 1, (n_nodes_ctx, 1))
    IA_init = 200.0 * np.random.uniform(0, 1, (n_nodes_ctx,))  # pA

    return (
        mufe_init,
        mufi_init,
        IA_init,
        seem_init,
        seim_init,
        seev_init,
        seiv_init,
        siim_init,
        siem_init,
        siiv_init,
        siev_init,
        rates_exc_init,
        rates_inh_init,
    )


def generateRandomICsThalamus(n_nodes_thal, seed=None):
    """
    Generates random initial conditions for the thalamus part of the interareal network

    :params n_nodes_thal: number of thalamic nodes.
    
    :returns: A tuple of length 15 representing initial state of the aln part of the model.
              All the elements are `n_nodes_thal` long numpy arrays.
              All the elements are vectors except for rates_tcr_init/rates_trn_init which are matrices.
    """

    np.random.seed(seed)  # TODO: For debug, remove when not needed. Ensures sanity check of identical output from thalamocortical to native aln and thalamus given same seed.

    voltage_tcr_init = np.random.uniform(-75, -50, (n_nodes_thal,))
    voltage_trn_init = np.random.uniform(-75, -50, (n_nodes_thal,))
    rates_tcr_init = np.random.uniform(0.0, 200.0, (n_nodes_thal, 1))
    rates_trn_init = np.random.uniform(0.0, 200.0, (n_nodes_thal, 1))
    Ca_init = np.ones((n_nodes_thal,)) * 2.4e-4
    h_T_t_init = np.zeros((n_nodes_thal,))
    h_T_r_init = np.zeros((n_nodes_thal,))
    m_h1_init = np.zeros((n_nodes_thal,))
    m_h2_init = np.zeros((n_nodes_thal,))
    s_et_init = np.zeros((n_nodes_thal,))
    s_gt_init = np.zeros((n_nodes_thal,))
    s_er_init = np.zeros((n_nodes_thal,))
    s_gr_init = np.zeros((n_nodes_thal,))
    ds_et_init = np.zeros((n_nodes_thal,))
    ds_gt_init = np.zeros((n_nodes_thal,))
    ds_er_init = np.zeros((n_nodes_thal,))
    ds_gr_init = np.zeros((n_nodes_thal,))
    
    return (
        voltage_tcr_init,
        voltage_trn_init,
        rates_tcr_init,
        rates_trn_init,
        Ca_init,
        h_T_t_init,
        h_T_r_init,
        m_h1_init,
        m_h2_init,
        s_et_init,
        s_gt_init,
        s_er_init,
        s_gr_init,
        ds_et_init,
        ds_gt_init,
        ds_er_init,
        ds_gr_init,
    )