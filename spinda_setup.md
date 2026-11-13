# Running training on Spinda

Three things we needs:
1. The `manifeel-isaaclab:latest` image (this doc).
2. The `manifeel` repo itself, cloned there (it's bind-mounted at runtime, not baked into the image --
   see `-v "${REPO_ROOT}:/workspace/manifeel"` in `scripts/run_maniflow_usb.sh`).
3. Whichever dataset(s) under `data/` you're training on. These are gitignored (`data/` in `.gitignore`),
   so `git clone` alone won't bring them.

`diffusion_policy/` and `maniflow/` do **not** need to exist on the other device at all -- they're
copied into the image at build time (see `docker/Dockerfile`), not bind-mounted, so whatever's already
integrated into the loaded image is all training needs.

### 1. Download the image

### 1. Export the image to a file

```bash
docker save -o manifeel-isaaclab.tar manifeel-isaaclab:latest
```

### 2. Checksum it

```bash
sha256sum manifeel-isaaclab.tar > manifeel-isaaclab.tar.sha256
```

Send both files. To be verified after downloading:
```bash
sha256sum -c manifeel-isaaclab.tar.sha256
```

### 3. Split if the transfer channel has a size cap

Some cloud storage/transfer services cap individual file size:
```bash
split -b 10G manifeel-isaaclab.tar manifeel-isaaclab.tar.part-
```
They reassemble before loading:
```bash
cat manifeel-isaaclab.tar.part-* > manifeel-isaaclab.tar
```

### 4. Load it on the other device

```bash
docker load -i manifeel-isaaclab.tar
```

This registers the image under the same tag (`manifeel-isaaclab:latest`) in that device's local
Docker image store.

### 5. Get the repo and dataset there too

```bash
git clone -b manifeel-isaaclab https://github.com/hisham246/manifeel.git
```


### 6. Verifying it worked

```bash
docker images manifeel-isaaclab:latest
cd manifeel && EXTRA_HYDRA_ARGS="logging.mode=disabled" ./scripts/run_maniflow_usb.sh
docker logs -f train_maniflow_usb
```
