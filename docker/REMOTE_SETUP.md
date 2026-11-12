# Running training on another device

`manifeel-isaaclab:latest` is a ~51GB image (Isaac Sim + Isaac Lab + manifeel/diffusion_policy/maniflow
baked in). If the other device is the **same GPU family** (e.g. also an RTX 5090), there's no need to
rebuild the Isaac Sim base image from scratch there -- just copy the already-built image over.

Three things the other device needs:
1. The `manifeel-isaaclab:latest` image (this doc).
2. The `manifeel` repo itself, cloned there (it's bind-mounted at runtime, not baked into the image --
   see `-v "${REPO_ROOT}:/workspace/manifeel"` in `scripts/run_maniflow_usb.sh`).
3. Whichever dataset(s) under `data/` you're training on. These are gitignored (`data/` in `.gitignore`),
   so `git clone` alone won't bring them.

`diffusion_policy/` and `maniflow/` do **not** need to exist on the other device at all -- they're
`COPY`'d into the image at build time (see `docker/Dockerfile`), not bind-mounted, so whatever's already
baked into the loaded image is all training needs.

## Option A -- you have SSH access to the other device

Use `scripts/setup_remote.sh`, which does all three steps in one go:

```bash
REMOTE=user@other-device SYNC_IMAGE=true ./scripts/setup_remote.sh
```

This clones/updates `manifeel` on the remote, `rsync`s the dataset (default:
`usb_quan_Aug05_contact_manifeel`, resumable via `-P` if the connection drops), and streams the image
directly with `docker save manifeel-isaaclab:latest | ssh ... docker load` -- no intermediate tarball
on either end. Drop `SYNC_IMAGE=true` if the image is already there or you're handling that transfer
separately.

## Option B -- no SSH access (handing the image to someone to download separately)

### 1. Export the image to a file

```bash
docker save -o manifeel-isaaclab.tar manifeel-isaaclab:latest
```

Skip gzip -- the image is mostly binary CUDA/Isaac Sim libraries, so compression barely shrinks it and
just burns CPU time on 51GB.

### 2. Checksum it

Worth doing given the size -- large files are more likely to get silently truncated/corrupted over a
flaky transfer:

```bash
sha256sum manifeel-isaaclab.tar > manifeel-isaaclab.tar.sha256
```

Send both files. The other person verifies after downloading:
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

This registers the image under the same name/tag (`manifeel-isaaclab:latest`) in that device's local
Docker image store -- byte-identical to the one built here, no rebuild involved. If that tag already
existed there, `docker load` just repoints it; the old layers become `<none>` and can be cleaned up with
`docker image prune`.

### 5. Get the repo and dataset there too

The image alone isn't enough -- `manifeel` is bind-mounted, not baked in:
```bash
git clone -b manifeel-isaaclab https://github.com/hisham246/manifeel.git
```

The dataset (e.g. `data/usb_quan_Aug05_contact_manifeel`, ~3.7GB) needs the same kind of file handoff as
the image, just smaller -- same channel, no special handling needed.

## Verifying it worked

On the other device:
```bash
docker images manifeel-isaaclab:latest   # same IMAGE ID as the source machine
cd manifeel && EXTRA_HYDRA_ARGS="logging.mode=disabled" ./scripts/run_maniflow_usb.sh
docker logs -f train_maniflow_usb
```
