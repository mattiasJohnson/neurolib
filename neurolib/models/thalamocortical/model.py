import numpy as np

from . import loadDefaultParams as dp
from . import timeIntegration as ti
from ..model import Model


class ThalamocorticalModel(Model):
    """
    Thalamocortical model with 80 cortical nodes and 2 thalamic nodes.
    """

    name = "thalamocortical"
    description = "Thalamocortical model with 80 cortical nodes and 2 thalamic nodes."

    init_vars = [
        # aln
        "rates_exc_init",
        "rates_inh_init",
        "mufe_init",
        "mufi_init",
        "IA_init",
        "seem_init",
        "seim_init",
        "siem_init",
        "siim_init",
        "seev_init",
        "seiv_init",
        "siev_init",
        "siiv_init",
        "mue_ou",
        "mui_ou",
        # thalamus
        "voltage_tcr_init",
        "voltage_trn_init",
        "rates_tcr_init",
        "rates_trn_init",
        "Ca_init",
        "h_T_t_init",
        "h_T_r_init",
        "m_h1_init",
        "m_h2_init",
        "s_et_init",
        "s_gt_init",
        "s_er_init",
        "s_gr_init",
        "ds_et_init",
        "ds_gt_init",
        "ds_er_init",
        "ds_gr_init",
    ]

    state_vars = [
        # aln
        "rates_exc",
        "rates_inh",
        "mufe",
        "mufi",
        "IA",
        "seem",
        "seim",
        "siem",
        "siim",
        "seev",
        "seiv",
        "siev",
        "siiv",
        "mue_ou",
        "mui_ou",
        # thalamus
        "voltage_tcr",
        "voltage_trn",
        "rates_tcr",
        "rates_trn",
        "Ca",
        "h_T_t",
        "h_T_r",
        "m_h1",
        "m_h2",
        "s_et",
        "s_gt",
        "s_er",
        "s_gr",
        "ds_et",
        "ds_gt",
        "ds_er",
        "ds_gr",
        # For debug
        "IA_array",
        "voltage_tcr_array",
        "voltage_trn_array",
        "thal_rowsum_array",
        "I_T_t_array",
        "I_T_r_array",
        "I_h_array",
        "Ca_array",
    ]
    output_vars = ["rates_exc", "rates_inh", "IA"] + ["voltage_tcr", "voltage_trn", "rates_tcr", "rates_trn"] + ["IA_array", "voltage_tcr_array", "voltage_trn_array", "thal_rowsum_array", "I_T_t_array", "I_T_r_array", "I_h_array", "Ca_array"]
    default_output = "rates_exc"
    input_vars = ["ext_exc_current", "ext_exc_rate"]
    default_input = "ext_exc_rate"

    def __init__(
        self,
        params=None,
        Cmat=None,
        Dmat=None,
        lookupTableFileName=None,
        seed=None,
        n_nodes_ctx=None,
        n_nodes_thal=None,
    ):
        """
        :param params: parameter dictionary of the model
        :param Cmat: Global connectivity matrix between the exc. aln and exc. thalamic nodes. Assumes first `n_nodes_ctx` indices are cortical
        :param Dmat: Distance matrix between all nodes (in mm)
        :param lookupTableFileName: Filename for precomputed transfer functions and tables
        :param seed: Random number generator seed
        :param simulateChunkwise: Chunkwise time integration (for lower memory use)
        """

        # Global attributes
        self.Cmat = Cmat  # Connectivity matrix
        self.Dmat = Dmat  # Delay matrix
        self.lookupTableFileName = lookupTableFileName  # Filename for aLN lookup functions
        self.seed = seed  # Random seed
        
        integration = ti.timeIntegration

        # load default parameters if none were given
        if params is None:
            params = dp.loadDefaultParams(
                Cmat=self.Cmat,
                Dmat=self.Dmat,
                lookupTableFileName=self.lookupTableFileName,
                seed=self.seed,
                n_nodes_ctx=n_nodes_ctx,
                n_nodes_thal=n_nodes_thal,
            )

        # Initialize base class Model
        super().__init__(integration=integration, params=params)

    def getMaxDelay(self):
        # compute maximum delay of model
        ndt_de = round(self.params["de"] / self.params["dt"])
        ndt_di = round(self.params["di"] / self.params["dt"])
        max_dmat_delay = super().getMaxDelay()
        return int(max(max_dmat_delay, ndt_de, ndt_di))