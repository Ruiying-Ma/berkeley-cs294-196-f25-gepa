# GEPA Run Instruction 
## Can't be late 
- Reference `gepa/adapters/cant_be_late_adapter` for the GEPA can't be late example 

1. **Install simulator dependencies and unpack the traces.**

   ```bash
   cd src/gepa/adapters/cant_be_late_adapter/simulator/
   uv sync --active
   mkdir -p data
   [ -d data/real ] || tar -xzf real_traces.tar.gz -C data
   mkdir -p ../../../../../exp
   [ -d ../../../../../exp/real ] || mv data/real ../../../../../exp/real
   ```

2. **Provide the API keys required by the prompts** (either export them manually or source a `.env`).

   ```bash
   export OPENAI_API_KEY=...
   export GEMINI_API_KEY=...
   ```

3. **Run example** 
```
cd ../../../../../
uv run --extra full src/gepa/adapters/cant_be_late_adapter/example.py
```