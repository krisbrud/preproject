import azureml
from azureml.core import Workspace
import mlflow

workspaces = Workspace.list("3165a1c1-fd45-4c8d-938e-0058c823f960")
ws = workspaces["aml-playground"]
playground_ws = ws[0]

## Construct AzureML MLFLOW TRACKING URI
def get_azureml_mlflow_tracking_uri(region, subscription_id, resource_group, workspace):
    return "azureml://{}.api.azureml.ms/mlflow/v1.0/subscriptions/{}/resourceGroups/{}/providers/Microsoft.MachineLearningServices/workspaces/{}".format(
        region, subscription_id, resource_group, workspace
    )


region = "northeurope"  ## example: westus
subscription_id = "3165a1c1-fd45-4c8d-938e-0058c823f960"  ## example: 11111111-1111-1111-1111-111111111111
resource_group = "aml-playground"  ## example: myresourcegroup
workspace = "aml-playground"  ## example: myworkspacename

MLFLOW_TRACKING_URI = get_azureml_mlflow_tracking_uri(
    region, subscription_id, resource_group, workspace
)

## Set the MLFLOW TRACKING URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

## Make sure the MLflow URI looks something like this:
## azureml://<REGION>.api.azureml.ms/mlflow/v1.0/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RESOURCE_GROUP>/providers/Microsoft.MachineLearningServices/workspaces/<AML_WORKSPACE_NAME>

print("MLFlow Tracking URI:", MLFLOW_TRACKING_URI)

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException, LocalTarget

# Choose a name for your cluster.
cluster_name = "preproject-cluster"
# instance_name = "cpuinstance1"
# cluster_name = instance_name


run_locally = True

if run_locally:
    compute_target = LocalTarget()
else:
    # Run in cloud

    # Check if we find a cluster with the name
    try:
        compute_target = ComputeTarget(workspace=playground_ws, name=cluster_name)
        print("Found existing compute target.")
    except ComputeTargetException:
        # Make a new one if an existing target is not found.
        print("Creating a new compute target...")
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="Standard_F4s_v2", max_nodes=1, idle_seconds_before_scaledown=300
        )

        # Create the cluster.
        compute_target = ComputeTarget.create(
            playground_ws, cluster_name, compute_config
        )

        compute_target.wait_for_completion(show_output=True)

# Use get_status() to get a detailed status for the current AmlCompute.
if not isinstance(compute_target, LocalTarget):
    print(compute_target.get_status().serialize())
