FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y git vim libgtk2.0-dev && \
    rm -rf /var/cache/apk/*

RUN pip --no-cache-dir install Cython

COPY requirements.txt /workspace

RUN pip --no-cache-dir install -r /workspace/requirements.txt

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

#RUN addgroup --gid "$USER_GID" "$USERNAME" && adduser --gid "$USER_GID" --uid "$USER_UID" --disabled-password --gecos "" "$USERNAME"
# Options for common package install script - SHA generated on release
ARG INSTALL_ZSH="false"
ARG UPGRADE_PACKAGES="true"
ARG COMMON_SCRIPT_SOURCE="https://raw.githubusercontent.com/microsoft/vscode-dev-containers/v0.122.1/script-library/common-debian.sh"
ARG COMMON_SCRIPT_SHA="da956c699ebef75d3d37d50569b5fbd75d6363e90b3f5d228807cff1f7fa211c"

# Configure apt and install packages
# RUN apt-get update && apt-get install --no-install-recommends -y \
#     python3-flake8 pylint python3-bandit python3-mypy python3-pycodestyle python3-pydocstyle \
#     python3-yapf python3-autopep8 pycodestyle pydocstyle yapf3 mypy flake8 pipx \
#     && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    #
    # Verify git, common tools / libs installed, add/modify non-root user, optionally install zsh
    && apt-get -y install --no-install-recommends curl ca-certificates 2>&1 \
    && curl -sSL  ${COMMON_SCRIPT_SOURCE} -o /tmp/common-setup.sh \
    && ([ "${COMMON_SCRIPT_SHA}" = "dev-mode" ] || (echo "${COMMON_SCRIPT_SHA} /tmp/common-setup.sh" | sha256sum -c -)) \
    && /bin/bash /tmp/common-setup.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" \
    && rm /tmp/common-setup.sh \
    && apt-get -y install --no-install-recommends build-essential build-essential cmake cppcheck valgrind clang lldb llvm gdb 2>&1 \
    #
    # Clean ups
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

