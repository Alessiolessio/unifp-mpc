"""Export model_5300 with the UniFP-Spot source checkout and check CPU action parity."""

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnx
import onnxruntime as ort
import torch


class StudentAction(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.encoder = policy.adaptation_encoder_module
        self.actor = policy.actor_body
        self.frame_width = policy.num_obs_now

    def forward(self, history):
        current = history[:, -self.frame_width:]
        return self.actor(torch.cat((current, self.encoder(history)), dim=-1))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../unifp-spot"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/encoder_actor.onnx"))
    args = parser.parse_args()
    if args.output.suffix != ".onnx":
        parser.error("output must have an .onnx suffix")
    source = args.source.resolve()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    from unifp_spot.learning.ppo_cse_pf.actor_critic import ActorCritic

    torch.set_num_threads(1)
    checkpoint = torch.load(source / "pretrained/spot_unifp/model_5300.pt",
                            map_location="cpu", weights_only=True)
    policy = ActorCritic(num_obs=2528, num_privileged_obs=459, num_obs_pred=12,
                         num_single_obs=79, num_actions=19).eval()
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    # A zero-padded first observation and distinct full histories exercise both input paths.
    generator = torch.Generator().manual_seed(5300)
    histories = torch.randn(8, 32, 79, generator=generator)
    histories[0, :-1].zero_()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as tmp:
        path = Path(tmp) / args.output.name
        torch.onnx.export(
            StudentAction(policy).eval(), (histories[:1].flatten(1),), str(path),
            input_names=["history"], output_names=["action"], opset_version=18, dynamo=False,
            dynamic_axes={"history": {0: "batch"}, "action": {0: "batch"}},
        )
        onnx.checker.check_model(str(path))
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        for inputs in (histories[:1].flatten(1), histories.flatten(1)):
            reference = policy.act_inference({"obs": inputs}, policy_info={}).numpy()
            (actual,) = session.run(["action"], {"history": inputs.numpy()})
            np.testing.assert_allclose(actual, reference, atol=1e-5, rtol=1e-5, equal_nan=False)
            error = np.max(np.abs(actual - reference))
            print(f"iteration {checkpoint['iter']}, batch {len(inputs)}, max |Δaction| {error:.3g}")
        path.replace(args.output)


if __name__ == "__main__":
    main()
