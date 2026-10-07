"""Entry point for GAIN_PHASE_PREREG.md: train_newobj.main() unchanged (task, recipe, eval.json), with the gain arms
GainRaw / GainPhase (model_gain_phase) added to train_newobj.ARMS. MapWM and NormStep are train_newobj's own classes
(asserted identical), so through this wrapper they are the LEAK arms (reproduction check in the prereg)."""
from mapformer import train_newobj
from mapformer.model_gain_phase import register

register(train_newobj.ARMS)

if __name__ == "__main__":
    train_newobj.main()
