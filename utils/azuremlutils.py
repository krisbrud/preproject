# Utils for using Azure Machine Learning
from azureml.core import Workspace


# Taken from https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-mlflow-cli-runs?tabs=mlflow#track-runs-from-your-local-machine
# Construct AzureML MLFlow tracking URI
def get_azureml_mlflow_tracking_uri(region, subscription_id, resource_group,
                                    workspace):
    """
    Gets a URI for tracking experiments and registering models with azureml-mlflow

    Params:
    region: str - Azure region - example \"northeurope\"
    subscription_id: str - Azure subscription ID - example \"11111111-1111-1111-1111-111111111111\"
    resource_group: str - Azure Resource group - example \"myresourcegroup\"
    workspace: str - Azure Machine Learning Workspace - example \"myworkspace\"
    """
    return "azureml://{}.api.azureml.ms/mlflow/v1.0/subscriptions/{}/resourceGroups/{}/providers/Microsoft.MachineLearningServices/workspaces/{}".format(
        region, subscription_id, resource_group, workspace)


def get_mlflow_uri_from_aml_workspace_config():
    """
    Try to get an mlflow_uri from a config (usually in '.azureml/config.json')
    """
    try:
        ws = Workspace.from_config()
        return ws.get_mlflow_tracking_uri()
    except:
        return None
