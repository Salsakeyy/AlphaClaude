FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

WORKDIR /app

# System deps for cmake build
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt pytest

# Copy source
COPY cpp_src/ cpp_src/
COPY py_src/ py_src/
COPY tests/ tests/
COPY CMakeLists.txt setup.py run_training.py ./

# Build C++ extension
RUN pip install -e .

# Verify build
RUN python -c "import alphaclaude_cpp as ac; g = ac.GameState(); print('Build OK:', g.fen())"
RUN python -m pytest tests/ -q

# Training writes here
RUN mkdir -p /app/checkpoints /app/logs

ENTRYPOINT ["python", "run_training.py"]
