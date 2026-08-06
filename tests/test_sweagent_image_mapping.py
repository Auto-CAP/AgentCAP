from agent_cap.agents.strategies_sweagent import _swebench_image


def test_docker_uses_official_swebench_image_name():
    assert _swebench_image("astropy__astropy-12907", "docker", "") == (
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    )


def test_custom_registry_keeps_official_image_suffix():
    assert _swebench_image(
        "django__django-12308", "docker", "registry.example/team"
    ) == (
        "registry.example/team/"
        "sweb.eval.x86_64.django_1776_django-12308:latest"
    )


def test_remote_deployments_use_fully_qualified_official_image():
    expected = (
        "docker.io/swebench/"
        "sweb.eval.x86_64.sympy_1776_sympy-18057:latest"
    )
    assert _swebench_image("sympy__sympy-18057", "modal", "") == expected
    assert _swebench_image("sympy__sympy-18057", "k8s", "") == expected
