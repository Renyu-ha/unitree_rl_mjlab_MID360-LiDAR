"""Create a clearly marked zero-action policy for simulation protocol checks."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def create(template, output, wheel_probe=False):
  import numpy as np
  import onnx
  from onnx import helper, numpy_helper, TensorProto
  from src.utils.go2w_deploy import write_deploy_yaml, validate_contract

  original=onnx.load(str(template),load_external_data=False)
  metadata={x.key:x.value for x in original.metadata_props}
  contract=json.loads(metadata["go2w_deploy_contract"])
  validate_contract(original,contract,metadata)
  actions=np.zeros((1,16),np.float32)
  if wheel_probe: actions[0,12:]=[0.05,0.1,0.15,0.2]
  graph=helper.make_graph([
    helper.make_node("Constant",[],["actions"],value=numpy_helper.from_array(actions))
  ],"go2w_simulation_smoke",[helper.make_tensor_value_info("obs",TensorProto.FLOAT,[1,contract["observation_dim"]])],
    [helper.make_tensor_value_info("actions",TensorProto.FLOAT,[1,16])])
  model=helper.make_model(graph,opset_imports=[helper.make_opsetid("",18)],ir_version=10)
  metadata["policy_kind"]="smoke_wheel_probe" if wheel_probe else "smoke_zero"
  helper.set_model_props(model,metadata)
  output=Path(output)
  exported=output/"exported"
  exported.mkdir(parents=True,exist_ok=True)
  onnx.save(model,exported/"policy.onnx")
  # The exporter writes params next to the policy; move it into G1's bundle layout.
  write_deploy_yaml(exported/"policy.onnx")
  params=output/"params"
  params.mkdir(exist_ok=True)
  (params/"deploy.yaml").write_bytes((exported/"params/deploy.yaml").read_bytes())
  (exported/"params/deploy.yaml").unlink()
  (exported/"params").rmdir()


if __name__=="__main__":
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--template",required=True,type=Path)
  parser.add_argument("--output",required=True,type=Path)
  parser.add_argument("--wheel-probe",action="store_true",help="Distinct wheel speeds for the simulation mapping check")
  args=parser.parse_args()
  create(args.template,args.output,args.wheel_probe)
