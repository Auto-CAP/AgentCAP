"""Start/stop/reset WebArena Docker services.

Downloads image tars (if needed) and starts 5 services:
  shopping (7770), shopping_admin (7780), forum (9999),
  gitlab (8023), wikipedia (8888)
"""

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("agent_cap.webarena.env")

SERVICES = {
    "shopping": {
        "image": "shopping_final_0712",
        "port": 7770,
        "internal_port": 80,
        "tar_url": "http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar",
    },
    "shopping_admin": {
        "image": "shopping_admin_final_0719",
        "port": 7780,
        "internal_port": 80,
        "tar_url": "http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar",
    },
    "forum": {
        "image": "postmill-populated-exposed-withimg",
        "port": 9999,
        "internal_port": 80,
        "tar_url": "http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar",
    },
    "gitlab": {
        "image": "gitlab-populated-final-port8023",
        "port": 8023,
        "internal_port": 8023,
        "tar_url": "http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar",
        "cmd": "/opt/gitlab/embedded/bin/runsvdir-start",
    },
}


def _run(cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout
    )


def start_services(hostname: str = "localhost") -> dict:
    urls = {}
    for name, svc in SERVICES.items():
        logger.info("Starting %s on port %d...", name, svc["port"])

        _run(f"docker rm -f {name}")

        cmd_parts = [
            "docker",
            "run",
            "--name",
            name,
            "-p",
            f"{svc['port']}:{svc['internal_port']}",
            "-d",
            svc["image"],
        ]
        if "cmd" in svc:
            cmd_parts.append(svc["cmd"])

        proc = subprocess.run(cmd_parts, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            logger.error("Failed to start %s: %s", name, proc.stderr[:300])
            continue

        urls[name] = f"http://{hostname}:{svc['port']}"

    logger.info("Waiting 60s for services to boot...")
    time.sleep(60)

    _configure_shopping(hostname)
    _configure_gitlab(hostname)

    return urls


def _configure_shopping(hostname: str):
    for name, port in [("shopping", 7770), ("shopping_admin", 7780)]:
        url = f"http://{hostname}:{port}"
        _run(
            f"docker exec {name} /var/www/magento2/bin/magento "
            f'setup:store-config:set --base-url="{url}"',
            timeout=30,
        )
        _run(
            f"docker exec {name} mysql -u magentouser -pMyPassword magentodb "
            f'-e \'UPDATE core_config_data SET value="{url}/" '
            f'WHERE path = "web/secure/base_url";\'',
            timeout=30,
        )
        _run(
            f"docker exec {name} /var/www/magento2/bin/magento cache:flush", timeout=30
        )

    _run(
        "docker exec shopping_admin php /var/www/magento2/bin/magento "
        "config:set admin/security/password_is_forced 0",
        timeout=30,
    )
    _run(
        "docker exec shopping_admin php /var/www/magento2/bin/magento "
        "config:set admin/security/password_lifetime 0",
        timeout=30,
    )


def _configure_gitlab(hostname: str):
    url = f"http://{hostname}:8023"
    _run("docker exec gitlab update-permissions", timeout=30)
    _run(
        f"docker exec gitlab sed -i "
        f"\"s|^external_url.*|external_url '{url}'|\" "
        f"/etc/gitlab/gitlab.rb",
        timeout=30,
    )
    _run("docker exec gitlab gitlab-ctl reconfigure", timeout=120)


def stop_services():
    for name in SERVICES:
        _run(f"docker stop {name}")
        _run(f"docker rm {name}")
    logger.info("All WebArena services stopped.")


def reset_services(hostname: str = "localhost") -> dict:
    stop_services()
    return start_services(hostname)


def check_services(hostname: str = "localhost") -> dict:
    import urllib.request

    status = {}
    checks = {
        "shopping": f"http://{hostname}:7770",
        "shopping_admin": f"http://{hostname}:7780",
        "forum": f"http://{hostname}:9999",
        "gitlab": f"http://{hostname}:8023",
    }
    for name, url in checks.items():
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            status[name] = resp.status
        except Exception:
            status[name] = 0
    return status
