
import streamlit as st
import torch
import time
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


st.set_page_config(page_title="KV Caching Explorer", layout="wide")
st.title(" KV Caching Techniques Explorer")
st.markdown("""
Enter a prompt and see how different KV caching strategies affect **speed**, **memory**, and **output quality**.
All methods use the **GPT-2** model for consistency.
""")

st.sidebar.header(" Settings")
user_prompt = st.sidebar.text_area(
    "Enter your prompt:",
    value="KV caching is a technique used in transformers to",
    height=100
)
max_new_tokens = st.sidebar.slider("Max New Tokens", 10, 100, 50)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    if use_cuda:
        model.to(device)
    return tokenizer, model

tokenizer, model = load_model()

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare input tensors
inputs = tokenizer(user_prompt, return_tensors="pt").to(device)

# Simulated performance modifiers (time_mult, memory_mult) based on research
STRATEGY_MODIFIERS = {
    "Quantized KV Cache": (1.10, 0.70),   # 10% slower, 30% less memory
    "Pruned KV Cache": (1.05, 0.60),      # 5% slower, 40% less memory
    "Offloaded KV Cache": (1.50, 0.50),   # 50% slower (CPU-GPU), 50% less GPU mem
    "LeanKV": (1.02, 0.65),               # Near-native speed, 35% less memory
    "MiKV": (1.08, 0.55),                 # Slight slowdown, 45% less memory
}

# Run button
if st.button(" Run All KV Caching Strategies"):
    results = []  # Will store: (name, time, memory, throughput, output_text)

    def run_inference(use_cache, strategy_name):
        if use_cuda:
            torch.cuda.empty_cache()
        start_time = time.time()
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                pad_token_id=tokenizer.eos_token_id
            )
        
        elapsed = time.time() - start_time
        memory_mb = torch.cuda.memory_allocated(device) / (1024 ** 2) if use_cuda else 0
        num_tokens = output.shape[-1]
        throughput = num_tokens / elapsed if elapsed > 0 else 0
        decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        results.append((strategy_name, elapsed, memory_mb, throughput, decoded_text))
        return elapsed, memory_mb, throughput, decoded_text

    with st.spinner("Running benchmarks..."):
        # 1. No KV Cache
        run_inference(use_cache=False, strategy_name="No KV Cache")
        
        # 2. Standard KV Cache
        run_inference(use_cache=True, strategy_name="KV Cache")
        
        # 3-7. Advanced strategies (simulated using KV Cache output + modifiers)
        for strat in ["Quantized KV Cache", "Pruned KV Cache", "Offloaded KV Cache", "LeanKV", "MiKV"]:
            # Extract baseline (KV Cache) result
            _, base_time, base_mem, _, decoded = results[1] 
            
            time_mult, mem_mult = STRATEGY_MODIFIERS[strat]
            sim_time = base_time * time_mult
            sim_mem = base_mem * mem_mult
            num_tokens = len(tokenizer.encode(decoded))
            sim_throughput = num_tokens / sim_time if sim_time > 0 else 0
            
            results.append((strat, sim_time, sim_mem, sim_throughput, decoded))

    # === DISPLAY RESULTS ===
    
    # 1. Generated outputs (expandable)
    st.subheader(" Generated Outputs")
    for name, _, _, _, text in results:
        with st.expander(f"✅ {name}"):
            st.write(text)

    # 2. Metrics table
    df = pd.DataFrame(results, columns=["Strategy", "Time (s)", "Memory (MB)", "Throughput (tok/s)", "Output"])
    df_metrics = df[["Strategy", "Time (s)", "Memory (MB)", "Throughput (tok/s)"]]
    
    st.subheader("📊 Performance Comparison")
    st.dataframe(df_metrics.style.format({
        "Time (s)": "{:.4f}",
        "Memory (MB)": "{:.2f}",
        "Throughput (tok/s)": "{:.2f}"
    }))

    # 3. Charts
    strategies = df_metrics["Strategy"].tolist()
    times = df_metrics["Time (s)"].tolist()
    memories = df_metrics["Memory (MB)"].tolist()
    throughputs = df_metrics["Throughput (tok/s)"].tolist()

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    axs[0].bar(strategies, times, color='skyblue')
    axs[0].set_title(" Inference Time (s)")
    axs[0].tick_params(axis='x', rotation=45)
    
    axs[1].bar(strategies, memories, color='lightgreen')
    axs[1].set_title(" Memory Usage (MB)")
    axs[1].tick_params(axis='x', rotation=45)
    
    axs[2].bar(strategies, throughputs, color='salmon')
    axs[2].set_title(" Throughput (tokens/sec)")
    axs[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)

# Footer note
st.markdown("---")
st.caption(
    " Note: Advanced KV techniques (Quantized, Pruned, etc.) are simulated for educational purposes. "
    "Real implementations require low-level modification of the attention cache (past_key_values)."
)
