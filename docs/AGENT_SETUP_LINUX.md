# HP Sensor Agent — Linux Setup

This page is the manual one-time setup for an HP laptop that will run the
PresenceMap v2 sensor agent. The agent uses an Atheros AR9271 USB Wi-Fi
adapter in **monitor mode** to capture CSI, plus the internal Wi-Fi for
SSH/Tailscale.

## 1. Distro + kernel

- **Ubuntu 22.04 LTS** is the supported target.
- **Kernel 5.15 (HWE up to 5.19)** is required: newer kernels break the
  ``ath9k`` CSI patch. Pin with:

  ```sh
  sudo apt install --install-recommends linux-generic-hwe-22.04
  ```

  Then check the running version:

  ```sh
  uname -r          # expect 5.15.x or 5.19.x
  ```

## 2. Patched ath9k driver (Atheros CSI Tool)

The CSI extraction lives in a patched ``ath9k`` driver. Build it from
[xieyaxiongfly/Atheros-CSI-Tool](https://github.com/xieyaxiongfly/Atheros-CSI-Tool):

```sh
sudo apt install build-essential bc libelf-dev linux-headers-$(uname -r) git
git clone https://github.com/xieyaxiongfly/Atheros-CSI-Tool.git
cd Atheros-CSI-Tool
# Build & install the patched driver against the running kernel. Follow the
# repo's README -- exact commands vary across forks.
make defconfig-ath9k-debug
make
sudo make install
```

After install, blacklist the stock ``ath9k_htc`` if any package re-installs
it, then ``sudo reboot``.

## 3. Stock firmware

Use stock ``htc_9271.fw`` -- no firmware patch is needed for the AR9271:

```sh
sudo apt install linux-firmware
```

## 4. Userspace recvCSI binary

The CSI tool ships a small C reader, ``recvCSI``. Build it:

```sh
git clone https://github.com/xieyaxiongfly/Atheros-CSI-Tool-UserSpace-APP.git
cd Atheros-CSI-Tool-UserSpace-APP/recvCSI
make
sudo cp recvCSI /usr/local/bin/
```

## 5. Passwordless sudo for iw

The agent calls ``sudo -n iw`` for scans and to read the noise floor.
Drop this file in:

```
# /etc/sudoers.d/presence
<your-user> ALL=(ALL) NOPASSWD: /usr/sbin/iw
```

```sh
sudo visudo -f /etc/sudoers.d/presence
sudo chmod 440 /etc/sudoers.d/presence
```

## 6. Per-boot bring-up

Run the helper after each reboot (or wire it to a systemd unit -- see
``tools/scripts/install_agent_service.sh``, lands in Stage 6):

```sh
sudo ./tools/scripts/atheros_csi_setup.sh --iface wlan1 --channel 6 --bw HT20
```

## 7. Verify

```sh
# preflight: webcam, NIC monitor mode, target SSID visible, etc.
presence-agent preflight --project ../data/survey_projects/apartment_test

# quick CSI capture (5 seconds)
presence-agent csi-test --project ../data/survey_projects/apartment_test \
                        --iface wlan1 --duration 5 --source atheros
```

Expected: ~150 frames over 5 s (30 Hz target), 56 subcarriers, sane
``amp_total_var`` and ``acf1`` values, ``loss_ratio`` near 0.

## 8. Run the agent

```sh
presence-agent run \
  --project ../data/survey_projects/apartment_test \
  --session smoketest_001 \
  --csi --csi-source atheros
```

The agent binds:

- ``tcp://0.0.0.0:5555`` -- WindowMsg + HealthMsg (msgpack)
- ``tcp://0.0.0.0:5556`` -- FrameMsg (multipart JPEG)

Point the Mac at this host -- see ``docs/ARCHITECTURE.md``.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| ``no working camera index`` | Internal webcam disabled or in use | check ``v4l2-ctl --list-devices``; reseat USB cam |
| ``target SSID not in N BSSes`` | Router off, hidden, or 5 GHz only | confirm target on the right band; agent listens on 2.4 GHz by default |
| ``csi-test`` returns 0 frames | Monitor mode mis-set, or wrong channel | re-run ``atheros_csi_setup.sh``; verify ``iw dev wlan1 info`` |
| ``sudo -n iw timed out`` | sudoers prompts for a password | step 5 above |
| ``rfkill blocked`` | RF kill-switch toggled | ``sudo rfkill unblock wifi`` |
