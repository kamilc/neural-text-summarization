build:
	podman build --memory 6G -t nlp:latest .
run:
	sudo podman run --privileged --memory 6G -p 8123:8123 -it nlp
bash:
	podman run --privileged -it nlp /bin/bash
