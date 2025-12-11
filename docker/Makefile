ACCOUNT := ctromanscoia

TAG := latest

IMAGE_NAME := histology_features

.DEFAULT_GOAL: build
build:
	docker build --platform=linux/amd64 -t $(ACCOUNT)/$(IMAGE_NAME):$(TAG) .
nc_build:
	docker build --no-cache --platform=linux/amd64 -t $(ACCOUNT)/$(IMAGE_NAME):$(TAG) .
run:
	docker run -it --rm -v $(PWD):/workspace --name $(IMAGE_NAME) $(ACCOUNT)/$(IMAGE_NAME):$(TAG)
run-gpu:
	docker run --gpus all -it --rm -v $(PWD):/workspace --name $(IMAGE_NAME) $(ACCOUNT)/$(IMAGE_NAME):$(TAG)
push:
	docker push $(ACCOUNT)/$(IMAGE_NAME):$(TAG)
singularity:
	echo "Converting $(ACCOUNT)/$(IMAGE_NAME) to singularity..."
	singularity pull --disable-cache --force $(IMAGE_NAME).simg docker://$(ACCOUNT)/$(IMAGE_NAME):$(TAG)
singularity-slurm:
	echo "Converting $(ACCOUNT)/$(IMAGE_NAME) to singularity..."
	srun --export "ALL" --error=error.txt --mem=256GB --cpus-per-task=48 --time=08:00:00 singularity pull --disable-cache --force $(IMAGE_NAME).simg docker://$(ACCOUNT)/$(IMAGE_NAME):$(TAG)
