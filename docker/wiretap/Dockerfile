# Stage 1: Clone the repository
FROM python:3.12-slim AS source

ARG WIRETAP_REPO=https://git.hmphin.space/wnbe/hmph-wiretap.git
ARG WIRETAP_BRANCH=main

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && git clone --depth 1 --branch "${WIRETAP_BRANCH}" \
       "${WIRETAP_REPO}" /tmp/wiretap \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=source /tmp/wiretap/companion/ ./companion/

EXPOSE 8787

CMD ["python", "/app/companion/wiretap_bridge.py"]
