import docker
import subprocess
import time

# Docker image name to run
image_name = "your-docker-image-name"


def run_docker_with_timeout(image_name, timeout_seconds=900):
    """Runs a Docker image with a given timeout and captures stdout.

    Args:
      image_name: Name of the Docker image to run.
      timeout_seconds: Maximum execution time in seconds.

    Returns:
      A tuple containing:
        - The stdout output of the container.
        - The return code (0 for success, non-zero for failure).
        - A boolean indicating if the container timed out.
    """

    client = docker.from_env()
    try:
        container = client.containers.run(image_name, detach=True)
        start_time = time.time()
        timeout = False

        # Wait for container to finish or timeout
        while container.status != "exited":
            time.sleep(1)
            if time.time() - start_time > timeout_seconds:
                timeout = True
                container.stop()
                break

        stdout = container.logs().decode("utf-8").strip()
        return_code = container.wait()["StatusCode"]
        container.remove()

        return stdout, return_code, timeout
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 1, False  # Return error values


# Run the Docker container with a 15-minute timeout
stdout, return_code, timed_out = run_docker_with_timeout(image_name, 15 * 60)

if timed_out:
    print(f"Container execution exceeded the time limit of 15 minutes.")
else:
    if return_code == 0:
        # Successful execution - process stdout
        try:
            data_line = stdout
            print(f"Data from container: {data_line}")
        except IndexError:
            print("Output does not contain a line of data as expected.")
    else:
        print(f"Container execution fail")
