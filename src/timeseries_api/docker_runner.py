import docker
import os
from docker.errors import ImageNotFound, APIError
from common.mainlog import MainLogger


class Evaluator:
    def __init__(
        self,
        image_name: str,
        bind_volume: str = "/tmp",
        logger: MainLogger = MainLogger(),
        trace: bool = False,
    ):
        self.client = docker.from_env()
        self.network = self.client.networks.create(
            "isolated_network", driver="bridge", internal=True
        )
        self.logger = logger
        self.image_name = image_name
        self.volume_configuration = {
            bind_volume: {
                "bind": "/tmp",
                "mode": "rw",
            },
        }
        self.device_requests = [
            docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
        ]
        self.env = {
            "SOME_ENV": os.environ.get("SOME_ENV"),
        }
        self.trace = trace

    def check_and_run_image(self, image_name: str, size_limit_gb: float = 4.0):
        client = self.client

        try:
            # Get image information without pulling
            image_info = client.images.get_registry_data(image_name)

            # Get the size in bytes and convert to GB
            size_gb = image_info.attrs["Size"] / (1024 * 1024 * 1024)

            print(f"Image: {image_name}")
            print(f"Size: {size_gb:.2f} GB")

            if size_gb > size_limit_gb:
                print(
                    f"Image size exceeds the limit of {size_limit_gb} GB. Not pulling or running."
                )
                return

            print("Size is within limit. Pulling and running the image...")

            # Pull the image
            image = client.images.pull(image_name)

            # Run the container
            container = client.containers.run(image_name, detach=True)

            print(f"Container ID: {container.id}")
            print("Container is now running.")

        except ImageNotFound:
            print(f"Image {image_name} not found in the registry.")
        except APIError as e:
            print(f"Error occurred while interacting with Docker API: {e}")

    def run_docker_container(
        self,
        job_type: str,
        request,
    ) -> dict:
        # Configure volume mounting
        volumes = self.volume_configuration
        # Configure GPU support
        device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

        command = f"{job_type} {request.to_args()}"
        self.logger.debug("command", command=command)

        # Run the container
        container = self.client.containers.run(
            self.image_name,
            command=command,
            volumes=volumes,
            device_requests=device_requests,
            environment=self.env,
            network=self.network,
            detach=True,  # Run in background
        )
        filepath = f"/tmp/{job_type}_output.json"
        filename = f"{job_type}_output.json"
        result = container.wait()
        while container.status == "created":
            time.sleep(10)
            container.reload()
        while container.status == "running":
            time.sleep(30)
            container.reload()

        self.logger.debug("container_run_complete")

        try:
            bits, stat = container.get_archive(filepath)
            with io.BytesIO() as file_data:
                for chunk in bits:
                    file_data.write(chunk)
                file_data.seek(0)
                with tarfile.open(fileobj=file_data) as tar:
                    content = tar.extractfile(filename).read().decode("utf-8")
                    container_results = json.loads(content)
                    self.logger.info(
                        "container_run_results",
                        details={
                            "filepath": filepath,
                            "content": content,
                            "result": result,
                            "container_id": container.id,
                        },
                    )
                    if not self.trace:
                        container.remove()
                    return container_results
        except Exception as e:
            self.logger.error("docker_error", error=e)
            if not self.trace:
                container.remove()
            return {"error": e}

    def prediction_score(
        self, request: EvaluateModelRequest
    ) -> Union[EvaluationScore, RunError]:
        try:
            eval_result = self.run_docker_container(
                job_type="prediction",
                request=request,
            )
            if "error" in eval_result:
                raise Exception(eval_result["error"])
            if eval_result["completed"] is False:
                raise Exception("completion internal error")
            score = eval_result["prediction"]
            return score
        except Exception as e:
            return RunError(error=str(e))


# Command to manually run evaluation
def entry():
    # add command line arguments for the ports of the two apis
    import argparse

    parser = argparse.ArgumentParser(description="Run a single evaluation instance")
    parser.add_argument(
        "--image", type=str, default="grader:latest", help="image to use"
    )
    parser.add_argument(
        "--repo_namespace", type=str, required=True, help="Repository namespace"
    )
    parser.add_argument("--repo_name", type=str, required=True, help="Repository name")
    parser.add_argument(
        "--chat_template_type", type=str, required=True, help="Chat template type"
    )
    parser.add_argument("--hash", type=str, required=True, help="Unique hash value")

    args = parser.parse_args()
    image_name = args.image
    print(f"running {image_name}")

    try:
        evaler = Evaluator(image_name=image_name, trace=True)
        prediction_result = evaler.prediction_score(req)
        print(f"prediction_result : {prediction_result}")
        if isinstance(prediction_result, RunError):
            raise Exception(prediction_result.error)

    except Exception as e:
        print(e)


if __name__ == "__main__":
    entry()
