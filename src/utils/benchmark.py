import torch
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import psutil
import gc


def get_model_size_mb(model):
    """Calculates the total size of model parameters and buffers in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024**2

def get_memory_usage_mb():
    """Returns the current RSS memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def _measure_callable_ms(fn, device, num_iterations=100, num_warmup=20):
    with torch.no_grad():
        for _ in range(num_warmup):
            fn()

        _synchronize(device)
        start = time.perf_counter()
        for _ in range(num_iterations):
            fn()
        _synchronize(device)
        end = time.perf_counter()

    return (end - start) / num_iterations * 1000


def _get_latency_ms(model, device, dummy_input, num_iterations=100, num_warmup=20):
    cpu_latency, gpu_latency, _ = run_single_benchmark(
        model,
        device,
        dummy_input,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )
    return gpu_latency if device.type == 'cuda' else cpu_latency


def _seed_model_for_benchmark(model_name, model, dummy_input):
    if model_name in ('ParallelHypernet', 'ParallelHypernet_NoGate', 'ParallelHypernet_SyncGate', 'SyncHypernet'):
        with torch.no_grad():
            feat = model._get_shallow(dummy_input)
            model.seed_experts(feat)
    elif model_name == 'PropertyPatch':
        with torch.no_grad():
            model.seed_properties(dummy_input)


def _average_profile_dicts(profiles):
    averaged = {}
    for key, value in profiles[0].items():
        if isinstance(value, dict):
            averaged[key] = {
                subkey: float(np.mean([profile[key][subkey] for profile in profiles]))
                for subkey in value
            }
        else:
            averaged[key] = float(np.mean([profile[key] for profile in profiles]))
    return averaged


def _profile_property_patch(model, device, dummy_input, num_iterations=100, num_warmup=20):
    if model.model_type == 'ConvNeXt':
        raise NotImplementedError('Component timing for PropertyPatch on ConvNeXt is not supported in the current benchmark harness.')

    features_ms = _measure_callable_ms(
        lambda: model._get_features(dummy_input),
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    with torch.no_grad():
        feat = model._get_features(dummy_input)
        base_logits = model._classify(feat)
        base_predictions = torch.argmax(base_logits, dim=1)
        deployment = model.compute_deployment_masks_from_features(dummy_input, feat, base_logits, base_predictions)
        allocation_mask = deployment['allocation_mask']

    indicator_ms = _measure_callable_ms(
        lambda: model.compute_deployment_masks_from_features(dummy_input, feat, base_logits, base_predictions),
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    patch_ms = _measure_callable_ms(
        lambda: model.apply_allocation_mask(feat, allocation_mask),
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    with torch.no_grad():
        patched_output = feat + model.apply_allocation_mask(feat, allocation_mask)

    classifier_ms = _measure_callable_ms(
        lambda: model._classify(patched_output),
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    end_to_end_ms = _get_latency_ms(
        model,
        device,
        dummy_input,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    critical_path_ms = features_ms + indicator_ms + patch_ms + classifier_ms
    return {
        'end_to_end_ms': float(end_to_end_ms),
        'critical_path_ms': float(critical_path_ms),
        'components': {
            'features_ms': float(features_ms),
            'indicator_allocation_ms': float(indicator_ms),
            'patch_ms': float(patch_ms),
            'classifier_ms': float(classifier_ms),
        },
    }


def _profile_sync_hypernet(model, device, dummy_input, num_iterations=100, num_warmup=20):
    shallow_ms = _measure_callable_ms(
        lambda: model._get_shallow(dummy_input),
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    with torch.no_grad():
        shallow_feat = model._get_shallow(dummy_input)

    deep_ms = _measure_callable_ms(
        lambda: model._get_deep(shallow_feat),
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )
    gate_ms = _measure_callable_ms(
        lambda: torch.min(torch.cdist(model._get_router_features(shallow_feat), model.expert_keys), dim=1).values,
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )
    hypernet_ms = _measure_callable_ms(
        lambda: model.hypernet(shallow_feat),
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    with torch.no_grad():
        deep_feat = model._get_deep(shallow_feat)
        gate_weight = model.hypernet(shallow_feat)

    if model.model_type == 'ConvNeXt':
        def classifier_call():
            reshaped_gate = gate_weight.view(-1, deep_feat.size(1), 1, 1)
            return model.classifier(deep_feat + reshaped_gate)
    else:
        deep_feat_flat = torch.flatten(deep_feat, 1) if model.model_type != 'ViT' else deep_feat

        def classifier_call():
            return model.classifier(deep_feat_flat + gate_weight)

    classifier_ms = _measure_callable_ms(
        classifier_call,
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    end_to_end_ms = _get_latency_ms(
        model,
        device,
        dummy_input,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    critical_path_ms = shallow_ms + deep_ms + hypernet_ms + gate_ms + classifier_ms
    return {
        'end_to_end_ms': float(end_to_end_ms),
        'critical_path_ms': float(critical_path_ms),
        'components': {
            'shallow_ms': float(shallow_ms),
            'deep_ms': float(deep_ms),
            'hypernet_ms': float(hypernet_ms),
            'gate_ms': float(gate_ms),
            'classifier_ms': float(classifier_ms),
        },
    }


def _profile_parallel_hypernet(model, device, dummy_input, num_iterations=100, num_warmup=20):
    shallow_ms = _measure_callable_ms(
        lambda: model._get_shallow(dummy_input),
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    with torch.no_grad():
        shallow_feat = model._get_shallow(dummy_input)

    deep_ms = _measure_callable_ms(
        lambda: model._get_deep(shallow_feat),
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )
    hypernet_ms = _measure_callable_ms(
        lambda: model.hypernet(shallow_feat),
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    def gate_call():
        if model.model_type != 'ViT':
            pooled = torch.nn.functional.adaptive_avg_pool2d(shallow_feat, (1, 1))
            x_flat = torch.flatten(pooled, 1)
        else:
            x_flat = shallow_feat[:, 0]
        dist = torch.cdist(x_flat, model.expert_keys)
        return (torch.min(dist, dim=1).values < model.threshold).float().view(-1, 1)

    gate_mode = getattr(model, 'gate_mode', 'parallel')
    if gate_mode != 'none':
        gate_ms = _measure_callable_ms(
            gate_call,
            device,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
        )
    else:
        gate_ms = 0.0

    with torch.no_grad():
        deep_feat = model._get_deep(shallow_feat)
        if gate_mode != 'none':
            patch = model.hypernet(shallow_feat)
            is_bug = gate_call()
            gate_weight = patch * is_bug
        else:
            gate_weight = model.hypernet(shallow_feat)

    if model.model_type == 'ConvNeXt':
        def classifier_call():
            reshaped_gate = gate_weight.view(-1, deep_feat.size(1), 1, 1)
            return model.classifier(deep_feat + reshaped_gate)
    else:
        deep_feat_flat = torch.flatten(deep_feat, 1) if model.model_type != 'ViT' else deep_feat

        def classifier_call():
            return model.classifier(deep_feat_flat + gate_weight)

    classifier_ms = _measure_callable_ms(
        classifier_call,
        device,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    end_to_end_ms = _get_latency_ms(
        model,
        device,
        dummy_input,
        num_iterations=num_iterations,
        num_warmup=num_warmup,
    )

    masked_branch_ms = hypernet_ms + gate_ms
    if gate_mode == 'sync':
        # gate runs serially after both streams join
        critical_path_ms = shallow_ms + max(deep_ms, hypernet_ms) + gate_ms + classifier_ms
    elif gate_mode == 'none':
        critical_path_ms = shallow_ms + max(deep_ms, hypernet_ms) + classifier_ms
    else:  # 'parallel'
        critical_path_ms = shallow_ms + max(deep_ms, masked_branch_ms) + classifier_ms
    return {
        'end_to_end_ms': float(end_to_end_ms),
        'critical_path_ms': float(critical_path_ms),
        'components': {
            'shallow_ms': float(shallow_ms),
            'deep_ms': float(deep_ms),
            'hypernet_ms': float(hypernet_ms),
            'gate_ms': float(gate_ms),
            'masked_branch_ms': float(masked_branch_ms),
            'classifier_ms': float(classifier_ms),
        },
    }


def _profile_model_components(model_name, model, device, dummy_input, num_iterations=100, num_warmup=20):
    if model_name == 'Vanilla':
        end_to_end_ms = _get_latency_ms(
            model,
            device,
            dummy_input,
            num_iterations=num_iterations,
            num_warmup=num_warmup,
        )
        return {
            'end_to_end_ms': float(end_to_end_ms),
            'critical_path_ms': float(end_to_end_ms),
            'components': {
                'forward_ms': float(end_to_end_ms),
            },
        }

    if model_name == 'PropertyPatch':
        return _profile_property_patch(model, device, dummy_input, num_iterations, num_warmup)
    if model_name == 'SyncHypernet':
        return _profile_sync_hypernet(model, device, dummy_input, num_iterations, num_warmup)
    if model_name in ('ParallelHypernet', 'ParallelHypernet_NoGate', 'ParallelHypernet_SyncGate'):
        return _profile_parallel_hypernet(model, device, dummy_input, num_iterations, num_warmup)

    raise ValueError(f'Unsupported model name for component profiling: {model_name}')

def run_single_benchmark(model, device, dummy_input, num_iterations=100, num_warmup=20):
    model.eval()
    
    # Clear memory before starting measurement to get a true baseline
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    mem_before = get_memory_usage_mb()
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
    start_cpu = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    
    if start_event:
        start_event.record()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            
    if end_event:
        end_event.record()
        torch.cuda.synchronize()
        
    end_cpu = time.perf_counter()
    mem_after = get_memory_usage_mb()
    
    cpu_latency = (end_cpu - start_cpu) / num_iterations * 1000 # in ms
    gpu_latency = start_event.elapsed_time(end_event) / num_iterations if start_event else cpu_latency
    
    peak_mem = mem_after - mem_before
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        
    return cpu_latency, gpu_latency, peak_mem

def run_latency_vs_bugs_all(cfg, device, instantiate_fn):
    print(f"\n[Latency vs Bugs Benchmark] Testing across models: {cfg.models_to_test}")
    num_experts_list = cfg.get('num_experts_list', [10, 20, 50, 100])
    
    results_data = {
        'num_experts': list(num_experts_list),
        'models': {}
    }
    
    dummy_input = torch.randn(cfg.get('batch_size', 1), 3, 224, 224).to(device)
    
    num_models = len(cfg.models_to_test)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), squeeze=False)
    axes = axes.flatten()
    
    for i, arch in enumerate(cfg.models_to_test):
        print(f"\n{'='*40}\nTesting Architecture: {arch}\n{'='*40}")
        
        models_config = [
            ('Vanilla', 'Vanilla'),
            ('PropertyPatch', 'PropertyPatch'),
            ('SyncHypernet', 'SyncHypernet'),
            ('ParallelHypernet', 'DynaPatch')
        ]
        
        results_data['models'][arch] = {m[0]: [] for m in models_config}
        num_iters = cfg.get('num_iterations', 100)
        
        for n in num_experts_list:
            print(f"\n---> Testing N = {n} Bugs")
            
            for m_key, m_display in models_config:
                # Instantiate ONE model
                model = instantiate_fn(arch, device, m_key, num_experts=n)
                
                # Seeding logic (if applicable)
                if m_key in ('ParallelHypernet', 'SyncHypernet'):
                    with torch.no_grad():
                        feat = model._get_shallow(dummy_input)
                        model.seed_experts(feat)
                elif m_key == 'PropertyPatch':
                    with torch.no_grad():
                        feat = model._get_features(dummy_input)
                        model.seed_properties(feat)
                
                # Benchmark
                latencies, memories = [], []
                for _ in range(cfg.num_trials):
                    _, l, m = run_single_benchmark(model, device, dummy_input, num_iterations=num_iters)
                    latencies.append(l); memories.append(m)
                
                l_mean, m_mean = np.mean(latencies), np.mean(memories)
                results_data['models'][arch][m_key].append(float(l_mean))
                
                # Storage & Reporting
                storage_str = ""
                if m_key == 'PropertyPatch':
                    storage = ((model.centers.nelement() * model.centers.element_size()) + 
                                (model.radii.nelement() * model.radii.element_size())) / 1024**2
                    storage += get_model_size_mb(model.patch_library)
                    storage_str = f" | Storage: {storage:.2f}MB"
                elif m_key == 'ParallelHypernet':
                    storage = (model.expert_keys.nelement() * model.expert_keys.element_size() / 1024**2) + get_model_size_mb(model.hypernet)
                    storage_str = f" | Storage: {storage:.2f}MB"
                
                print(f"  {m_display:<16} | {l_mean:.2f} ms | Mem: {m_mean:.1f}MB{storage_str}")
                
                # AGGRESSIVE CLEANUP
                del model
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
        ax = axes[i]
        ax.plot(num_experts_list, results_data['models'][arch]['Vanilla'], 'k--', label='Vanilla')
        ax.plot(num_experts_list, results_data['models'][arch]['PropertyPatch'], 'r-s', label='PropertyPatch')
        ax.plot(num_experts_list, results_data['models'][arch]['SyncHypernet'], 'g-o', label='SyncHypernet')
        ax.plot(num_experts_list, results_data['models'][arch]['ParallelHypernet'], 'b-*', markersize=10, label='DynaPatch')
        ax.set_title(f'Latency ({arch})'); ax.legend()

    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.tight_layout(); plt.savefig(f"results/latency_vs_bugs_{timestamp}.png"); plt.close()
    with open(f"results/latency_vs_bugs_{timestamp}.json", 'w') as f: json.dump(results_data, f, indent=4)

def run_storage_benchmark(cfg, device, instantiate_fn):
    print(f"\n[Storage Benchmark] O(1) vs O(N) Scalability Test across models: {cfg.models_to_test}")
    num_experts_list = cfg.get('num_experts_list', [10, 50, 100, 500, 1000])
    results_data = {'num_experts': list(num_experts_list), 'models': {}}
    num_models = len(cfg.models_to_test)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), squeeze=False)
    axes = axes.flatten()
    for i, arch in enumerate(cfg.models_to_test):
        results_data['models'][arch] = {'PropertyPatch (O(N))': [], 'DynaPatch (O(1))': []}
        for n in num_experts_list:
            # Test PropertyPatch
            m_patch = instantiate_fn(arch, device, 'PropertyPatch', num_experts=n)
            sync_overhead_mb = ((m_patch.centers.nelement() * m_patch.centers.element_size()) + 
                                (m_patch.radii.nelement() * m_patch.radii.element_size())) / 1024**2
            sync_overhead_mb += get_model_size_mb(m_patch.patch_library)
            results_data['models'][arch]['PropertyPatch (O(N))'].append(sync_overhead_mb)
            del m_patch
            
            # Test DynaPatch
            m_para = instantiate_fn(arch, device, 'ParallelHypernet', num_experts=n)
            para_overhead_mb = (m_para.expert_keys.nelement() * m_para.expert_keys.element_size() / 1024**2)
            para_overhead_mb += get_model_size_mb(m_para.hypernet)
            results_data['models'][arch]['DynaPatch (O(1))'].append(para_overhead_mb)
            del m_para
            
            # Cleanup
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        ax = axes[i]
        ax.plot(num_experts_list, results_data['models'][arch]['PropertyPatch (O(N))'], 'r--', label='PropertyPatch (O(N))')
        ax.plot(num_experts_list, results_data['models'][arch]['DynaPatch (O(1))'], 'b-', label='DynaPatch (O(1))')
        ax.set_title(f'Storage Overheads ({arch})'); ax.set_ylabel('Size (MB)'); ax.legend()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.tight_layout(); plt.savefig(f"results/storage_{timestamp}.png"); plt.close()
    with open(f"results/storage_{timestamp}.json", 'w') as f: json.dump(results_data, f, indent=4)

def run_latency_benchmark(cfg, device, instantiate_fn):
    print(f"\n[Latency vs Batch Size Benchmark] Testing across models: {cfg.models_to_test}")
    batch_sizes = cfg.get('batch_sizes', [1, 8, 32, 64, 128])
    results_data = {'batch_sizes': list(batch_sizes), 'models': {}}
    num_models = len(cfg.models_to_test); fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), squeeze=False); axes = axes.flatten()
    for i, arch in enumerate(cfg.models_to_test):
        models_config = [
            ('Vanilla', 'Vanilla'),
            ('PropertyPatch', 'PropertyPatch'),
            ('SyncHypernet', 'SyncHypernet'),
            ('ParallelHypernet', 'DynaPatch')
        ]
        results_data['models'][arch] = {m[0]: [] for m in models_config}
        default_num_experts = cfg.get('num_experts', 50)
        
        for m_key, m_display in models_config:
            model = instantiate_fn(arch, device, m_key, num_experts=default_num_experts)
            for bs in batch_sizes:
                dummy_input = torch.randn(bs, 3, 224, 224).to(device)
                latency_ms = _get_latency_ms(
                    model,
                    device,
                    dummy_input,
                    num_iterations=cfg.get('num_iterations', 100),
                )
                results_data['models'][arch][m_key].append(float(latency_ms))
            
            # Cleanup after each model variant
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        ax = axes[i]; x = np.arange(len(batch_sizes)); width = 0.2
        ax.bar(x - 1.5*width, results_data['models'][arch]['Vanilla'], width, label='Vanilla')
        ax.bar(x - 0.5*width, results_data['models'][arch]['PropertyPatch'], width, label='PropertyPatch')
        ax.bar(x + 0.5*width, results_data['models'][arch]['SyncHypernet'], width, label='SyncHypernet')
        ax.bar(x + 1.5*width, results_data['models'][arch]['ParallelHypernet'], width, label='DynaPatch')
        ax.set_xticks(x); ax.set_xticklabels(batch_sizes); ax.set_title(f'Batch Size ({arch})'); ax.legend()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.tight_layout(); plt.savefig(f"results/latency_vs_bs_{timestamp}.png"); plt.close()
    with open(f"results/latency_vs_bs_{timestamp}.json", 'w') as f: json.dump(results_data, f, indent=4)

def run_latency_vs_resolution(cfg, device, instantiate_fn):
    print(f"\n[Latency vs Resolution Benchmark] Testing across models: {cfg.models_to_test}")
    resolutions = cfg.get('resolutions_to_test', [224, 384, 512, 768, 1024])
    results_data = {'resolutions': list(resolutions), 'models': {}}
    num_models = len(cfg.models_to_test); fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), squeeze=False); axes = axes.flatten()
    for i, arch in enumerate(cfg.models_to_test):
        if arch == 'vit': continue
        models_config = [
            ('Vanilla', 'Vanilla (GPU)', 'k--'),
            ('PropertyPatch', 'PropertyPatch (GPU)', 'r-s'),
            ('SyncHypernet', 'SyncHypernet (GPU)', 'g-o'),
            ('ParallelHypernet', 'DynaPatch (GPU)', 'b-*')
        ]
        results_data['models'][arch] = {m[1]: [] for m in models_config}
        default_num_experts = cfg.get('num_experts', 50)
        
        for m_key, m_display, m_style in models_config:
            model = instantiate_fn(arch, device, m_key, num_experts=default_num_experts)
            for res in resolutions:
                dummy_input = torch.randn(1, 3, res, res).to(device)
                latency_ms = _get_latency_ms(
                    model,
                    device,
                    dummy_input,
                    num_iterations=cfg.get('num_iterations', 100),
                )
                results_data['models'][arch][m_display].append(float(latency_ms))
            
            # Plotting this model
            ax = axes[i]
            ax.plot(resolutions, results_data['models'][arch][m_display], m_style, label=m_display.split(' ')[0])
            
            # Cleanup after each model variant
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        ax.set_title(f'Resolution ({arch})'); ax.legend()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.tight_layout(); plt.savefig(f"results/latency_vs_res_{timestamp}.png"); plt.close()
    with open(f"results/latency_vs_res_{timestamp}.json", 'w') as f: json.dump(results_data, f, indent=4)


def run_component_timing_benchmark(cfg, device, instantiate_fn):
    print(f"\n[Component Timing Benchmark] Testing across models: {cfg.models_to_test}")
    batch_size = cfg.get('batch_size', 1)
    input_resolution = cfg.get('input_resolution', 224)
    num_experts = cfg.get('num_experts', 50)
    num_iterations = cfg.get('num_iterations', 100)
    num_warmup = cfg.get('num_warmup', 20)

    results_data = {
        'batch_size': int(batch_size),
        'input_resolution': int(input_resolution),
        'num_experts': int(num_experts),
        'models': {},
    }

    models_config = [
        ('Vanilla', 'Vanilla'),
        ('PropertyPatch', 'PropertyPatch'),
        ('SyncHypernet', 'SyncHypernet'),
        ('ParallelHypernet', 'DynaPatch'),
    ]
    plotted_results = {}

    num_models = len(cfg.models_to_test)
    fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6), squeeze=False)
    axes = axes.flatten()

    for axis_index, arch in enumerate(cfg.models_to_test):
        print(f"\n{'=' * 40}\nProfiling Architecture: {arch}\n{'=' * 40}")
        dummy_input = torch.randn(batch_size, 3, input_resolution, input_resolution).to(device)
        results_data['models'][arch] = {}
        plotted_results[arch] = []

        for model_name, display_name in models_config:
            model = instantiate_fn(arch, device, model_name, num_experts=num_experts)
            _seed_model_for_benchmark(model_name, model, dummy_input)

            try:
                profiles = []
                for _ in range(cfg.get('num_trials', 1)):
                    profiles.append(
                        _profile_model_components(
                            model_name,
                            model,
                            device,
                            dummy_input,
                            num_iterations=num_iterations,
                            num_warmup=num_warmup,
                        )
                    )
                profile = _average_profile_dicts(profiles)
                results_data['models'][arch][model_name] = profile
                plotted_results[arch].append((model_name, display_name))
                print(
                    f"  {display_name:<16} | end-to-end: {profile['end_to_end_ms']:.2f} ms "
                    f"| critical-path: {profile['critical_path_ms']:.2f} ms"
                )
            except NotImplementedError as exc:
                results_data['models'][arch][model_name] = {
                    'skipped': True,
                    'reason': str(exc),
                }
                print(f"  {display_name:<16} | skipped: {exc}")

            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        vanilla_profile = results_data['models'][arch].get('Vanilla')
        if vanilla_profile and 'end_to_end_ms' in vanilla_profile:
            vanilla_latency = vanilla_profile['end_to_end_ms']
            for model_name, _ in plotted_results[arch]:
                end_to_end_ms = results_data['models'][arch][model_name]['end_to_end_ms']
                results_data['models'][arch][model_name]['latency_overhead_ratio'] = float(
                    (end_to_end_ms - vanilla_latency) / vanilla_latency if vanilla_latency > 0 else 0.0
                )

        axis = axes[axis_index]
        labels = [display_name for _, display_name in plotted_results[arch]]
        end_to_end_values = [results_data['models'][arch][model_name]['end_to_end_ms'] for model_name, _ in plotted_results[arch]]
        critical_path_values = [results_data['models'][arch][model_name]['critical_path_ms'] for model_name, _ in plotted_results[arch]]
        positions = np.arange(len(labels))
        width = 0.36
        if labels:
            axis.bar(positions - width / 2, end_to_end_values, width, label='Measured end-to-end')
            axis.bar(positions + width / 2, critical_path_values, width, label='Component critical-path')
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=20)
            axis.set_ylabel('Latency (ms)')
            axis.set_title(f'Component Timing ({arch})')
            axis.legend()
        else:
            axis.set_axis_off()

    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.tight_layout(); plt.savefig(f"results/component_timing_{timestamp}.png"); plt.close()
    with open(f"results/component_timing_{timestamp}.json", 'w') as f: json.dump(results_data, f, indent=4)


def run_gate_comparison_benchmark(cfg, device, instantiate_fn):
    """Compare no-gate / sync-gate / parallel-gate DynaPatch variants against Vanilla and
    SyncHypernet to isolate the cost and benefit of each gating mechanism."""
    print(f"\n[Gate Comparison Benchmark] Testing across models: {cfg.models_to_test}")
    batch_size = cfg.get('batch_size', 1)
    input_resolution = cfg.get('input_resolution', 224)
    num_experts = cfg.get('num_experts', 50)
    num_iterations = cfg.get('num_iterations', 200)
    num_warmup = cfg.get('num_warmup', 50)

    models_config = [
        ('Vanilla',                   'Vanilla'),
        ('SyncHypernet',              'Sync+Gate'),
        ('ParallelHypernet_NoGate',   'Parallel (no gate)'),
        ('ParallelHypernet_SyncGate', 'Parallel (sync gate)'),
        ('ParallelHypernet',          'Parallel (par. gate)'),
    ]

    results_data = {
        'batch_size': int(batch_size),
        'input_resolution': int(input_resolution),
        'num_experts': int(num_experts),
        'models': {},
    }

    num_arch = len(cfg.models_to_test)
    fig, axes = plt.subplots(1, num_arch, figsize=(8 * num_arch, 6), squeeze=False)
    axes = axes.flatten()

    for axis_index, arch in enumerate(cfg.models_to_test):
        print(f"\n{'=' * 40}\nGate Comparison: {arch}\n{'=' * 40}")
        dummy_input = torch.randn(batch_size, 3, input_resolution, input_resolution).to(device)
        results_data['models'][arch] = {}

        for model_name, display_name in models_config:
            model = instantiate_fn(arch, device, model_name, num_experts=num_experts)
            _seed_model_for_benchmark(model_name, model, dummy_input)

            latencies = [
                _get_latency_ms(model, device, dummy_input,
                                num_iterations=num_iterations, num_warmup=num_warmup)
                for _ in range(cfg.get('num_trials', 3))
            ]
            mean_ms = float(np.mean(latencies))
            std_ms = float(np.std(latencies))
            results_data['models'][arch][model_name] = {
                'display_name': display_name,
                'end_to_end_ms': mean_ms,
                'std_ms': std_ms,
            }
            print(f"  {display_name:<24} | {mean_ms:.3f} ± {std_ms:.3f} ms")

            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        vanilla_ms = results_data['models'][arch]['Vanilla']['end_to_end_ms']
        for model_name, _ in models_config:
            end_ms = results_data['models'][arch][model_name]['end_to_end_ms']
            results_data['models'][arch][model_name]['overhead_ratio'] = float(
                (end_ms - vanilla_ms) / vanilla_ms if vanilla_ms > 0 else 0.0
            )

        axis = axes[axis_index]
        labels = [display_name for _, display_name in models_config]
        values = [results_data['models'][arch][m]['end_to_end_ms'] for m, _ in models_config]
        colors = ['#888888', '#4c72b0', '#55a868', '#dd8452', '#c44e52']
        bars = axis.bar(np.arange(len(labels)), values, color=colors)
        axis.axhline(vanilla_ms, color='black', linestyle='--', linewidth=0.8,
                     label=f'Vanilla ({vanilla_ms:.2f} ms)')
        for bar, val in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, val + val * 0.01,
                      f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        axis.set_xticks(np.arange(len(labels)))
        axis.set_xticklabels(labels, rotation=20, ha='right')
        axis.set_ylabel('Latency (ms)')
        axis.set_title(f'Gate Mode Comparison ({arch})')
        axis.legend()

    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(f"results/gate_comparison_{timestamp}.png")
    plt.close()
    with open(f"results/gate_comparison_{timestamp}.json", 'w') as f:
        json.dump(results_data, f, indent=4)


# ResNet50: depth → (shallow_dim, human label)
_RESNET_SPLIT_DEPTHS = {
    4: (64,   'depth=4\n(before layer1)'),
    5: (256,  'depth=5\n(after layer1)'),
    6: (512,  'depth=6\n(after layer2)'),
    7: (1024, 'depth=7\n(after layer3)'),
}


def run_split_depth_benchmark(cfg, device, instantiate_fn):
    """Sweep the shallow/deep split point in ParallelHypernet (ResNet50 only).

    For each split depth we measure:
      - shallow_ms  (stream0, always serial)
      - deep_ms     (stream2)
      - hypernet+gate_ms  (stream1)
      - end_to_end_ms
      - theoretical critical path = shallow + max(deep, hypernet+gate) + cls

    This reveals the optimal depth where stream1 ≈ stream2 (maximum parallelism).
    """
    arch = 'resnet50'
    num_experts = cfg.get('num_experts', 50)
    num_iterations = cfg.get('num_iterations', 200)
    num_warmup = cfg.get('num_warmup', 50)
    num_trials = cfg.get('num_trials', 3)
    batch_size = cfg.get('batch_size', 1)
    input_resolution = cfg.get('input_resolution', 224)

    print(f"\n[Split-Depth Benchmark] ResNet50, batch={batch_size}, res={input_resolution}, "
          f"n_experts={num_experts}, iters={num_iterations}")

    dummy_input = torch.randn(batch_size, 3, input_resolution, input_resolution).to(device)

    depths = sorted(_RESNET_SPLIT_DEPTHS.keys())
    results_data = {
        'arch': arch, 'batch_size': int(batch_size),
        'input_resolution': int(input_resolution),
        'num_experts': int(num_experts),
        'depths': {},
    }

    for depth in depths:
        shallow_dim, label = _RESNET_SPLIT_DEPTHS[depth]
        print(f"\n  --- split_depth={depth}  shallow_dim={shallow_dim} ---")

        model = instantiate_fn(
            arch, device, 'ParallelHypernet',
            num_experts=num_experts,
            split_depth=depth,
            shallow_dim_override=shallow_dim,
        )
        _seed_model_for_benchmark('ParallelHypernet', model, dummy_input)

        profiles = []
        for _ in range(num_trials):
            profiles.append(
                _profile_parallel_hypernet(
                    model, device, dummy_input,
                    num_iterations=num_iterations,
                    num_warmup=num_warmup,
                )
            )
        profile = _average_profile_dicts(profiles)

        results_data['depths'][depth] = {
            'shallow_dim': shallow_dim,
            'label': label,
            **profile,
        }

        c = profile['components']
        print(f"    shallow={c['shallow_ms']:.2f}ms  deep={c['deep_ms']:.2f}ms  "
              f"hypernet={c['hypernet_ms']:.2f}ms  gate={c['gate_ms']:.2f}ms  "
              f"stream1={c['masked_branch_ms']:.2f}ms  | "
              f"end-to-end={profile['end_to_end_ms']:.2f}ms  "
              f"critical-path={profile['critical_path_ms']:.2f}ms")

        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = [_RESNET_SPLIT_DEPTHS[d][1] for d in depths]
    shallow_vals   = [results_data['depths'][d]['components']['shallow_ms']       for d in depths]
    deep_vals      = [results_data['depths'][d]['components']['deep_ms']           for d in depths]
    stream1_vals   = [results_data['depths'][d]['components']['masked_branch_ms']  for d in depths]
    e2e_vals       = [results_data['depths'][d]['end_to_end_ms']                   for d in depths]
    cp_vals        = [results_data['depths'][d]['critical_path_ms']                for d in depths]
    x = np.arange(len(depths))
    w = 0.25

    ax1.bar(x - w, shallow_vals, w, label='shallow_ms')
    ax1.bar(x,     deep_vals,    w, label='deep_ms (stream2)')
    ax1.bar(x + w, stream1_vals, w, label='hypernet+gate_ms (stream1)')
    ax1.set_xticks(x); ax1.set_xticklabels(labels)
    ax1.set_ylabel('ms'); ax1.set_title('Component latency vs split depth (ResNet50)')
    ax1.legend()

    ax2.plot(depths, e2e_vals,  'b-o', label='Measured end-to-end')
    ax2.plot(depths, cp_vals,   'r--s', label='Theoretical critical path')
    ax2.set_xticks(depths)
    ax2.set_xlabel('split_depth (# ResNet50 children in shallow)')
    ax2.set_ylabel('ms'); ax2.set_title('End-to-end latency vs split depth (ResNet50)')
    ax2.legend()

    os.makedirs('results', exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.tight_layout()
    plt.savefig(f"results/split_depth_{timestamp}.png")
    plt.close()
    with open(f"results/split_depth_{timestamp}.json", 'w') as f:
        json.dump(results_data, f, indent=4)
    print(f"\n[Split-Depth] saved results/split_depth_{timestamp}.{{png,json}}")
