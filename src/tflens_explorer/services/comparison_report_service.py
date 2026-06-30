import math
from tflens_explorer.core.snapshot_types import Snapshot, SNAPSHOT_PATH, SNAPSHOT_DATA_PATH
from tflens_explorer.core.comparison_types import HeadSimilarity, CacheActivationDifferences

def display_cache_activation_summary(data: CacheActivationDifferences, header=False) -> None:

    print(
        f"    {'A/B':<4}"
        f"{'hook_name':<36}"
        f"{'shape':>15}"
        f"{'min':>13}"
        f"{'max':>13}"
        f"{'mean':>13}"
        f"{'mean_abs_diff':>16}"
        f"{'cos_sim':>16}"
    )

    print(
        f"    {'A:':<4}"
        f"{hook1:<35} "
        f"{shape1:>15} "
        f"{minimum1:>12.4f} "
        f"{maximum1:>12.4f} "
        f"{mean1_str:>12}"
        f"{mean_abs_diff_str:>16}"
        f"{cos_similarity_str:>16}"
    )
    

    return

def angular_change_per_head(filename: str) -> None:
    filepath = SNAPSHOT_DATA_PATH / filename
    if not filepath.is_file():
        print(f"File not found: {filepath}")
        raise SystemExit(1)

    # ── read CSV ──────────────────────────────────────────────────────
    rows: list[tuple[str, int, float]] = []     # (hook, head, cos_sim)
    seen_hooks: list[str] = []
    seen_heads: set[int] = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            hook = parts[0].strip()
            head = int(parts[2].strip())
            cos_sim = float(parts[3].strip())
            rows.append((hook, head, cos_sim))
            if hook not in seen_hooks:
                seen_hooks.append(hook)
            seen_heads.add(head)

    print("Angular similarity by head (top 10)")

    # Descending List 
    print(
        f"{'    ':<4} "
        f"{'hook_name':<36} "
        f"{'head':>5} "
        f"{'angle':>8}"
    )

    if not rows:
        print("No data rows found.")
        return

    rows.sort(key=lambda t: t[-1])

    row_count = 0
    for row in rows:
        if ".o." in row[0] or ".v." in row[0]:
            continue
        row_count +=1
        layer = row[0]
        head = row[1]
        angle_rad = math.acos(row[2])
        angle_deg = math.degrees(angle_rad)

        print(
            f"{'    ':<4} "
            f"{layer:<36} "
            f"{head:>5d} "
            f"{angle_deg:>8.4f}"
        )
        if row_count == 10:
            break

    # Ascending List 
    print()
    print(
        f"{'    ':<4} "
        f"{'hook_name':<36} "
        f"{'head':>5} "
        f"{'angle':>8}"
    )

    rows.sort(key=lambda t: t[-1], reverse=True)

    row_count = 0
    for row in rows:
        if ".o." in row[0] or ".v." in row[0]:
            continue
        row_count +=1
        layer = row[0]
        head = row[1]
        angle_rad = math.acos(row[2])
        angle_deg = math.degrees(angle_rad)

        print(
            f"{'    ':<4} "
            f"{layer:<36} "
            f"{head:>5d} "
            f"{angle_deg:>8.4f}"
        )
        if row_count == 10:
            break

    return

def plot_cosine_chart(filename: str) -> None:
    """Read cosine similarity per-head CSV data and plot a line-segment chart.

    The CSV is expected at ``snapshots/data/<filename>`` with the format::

        hook1, hook2, head, cosine_similarity

    For each attention head a piecewise-linear path is drawn.  Each segment
    spans one unit on the x-axis and its *slope* is set to
    ``arccos(cosine_similarity)``.  Cosine similarity close to 1 therefore
    produces a flat segment, while a low similarity produces a steep upward
    segment.

    The output image is saved to ``snapshots/data/<filename>.png``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    filepath = SNAPSHOT_DATA_PATH / filename
    if not filepath.is_file():
        print(f"File not found: {filepath}")
        return

    # ── read CSV ──────────────────────────────────────────────────────
    rows: list[tuple[str, int, float]] = []     # (hook, head, cos_sim)
    seen_hooks: list[str] = []
    seen_heads: set[int] = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            hook = parts[0].strip()
            head = int(parts[2].strip())
            cos_sim = float(parts[3].strip())
            rows.append((hook, head, cos_sim))
            if hook not in seen_hooks:
                seen_hooks.append(hook)
            seen_heads.add(head)

    if not rows:
        print("No data rows found.")
        return

    # Build (hook_index, cos_sim) per head, preserving file order
    hook_index_of = {h: i for i, h in enumerate(seen_hooks)}
    heads: dict[int, list[tuple[int, float]]] = {h: [] for h in seen_heads}
    for hook, head, cos_sim in rows:
        heads[head].append((hook_index_of[hook], cos_sim))

    # ── draw ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    n_heads = len(heads)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_heads))
    
    for (head, segments), color in zip(
        sorted(heads.items()), colors
    ):
        x = 0.0
        y = 0.0
        xs: list[float] = [x]
        ys: list[float] = [y]
        for _hook_idx, cos_sim in sorted(segments):
            angle = math.acos(max(-1.0, min(1.0, cos_sim)))  # clamp to [-1, 1]
            x += 1.0
            y += angle
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, color=color, label=f"Head {head}")

    ax.set_xlabel("Hook index (each unit = one hook)")
    ax.set_ylabel("Cumulative arccos(cosine similarity)")
    ax.set_title(f"Cosine similarity change per head — {filename}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

    # Tick labels – abbreviated hook names
    index = 0
    for item in seen_hooks:
        seen_hooks[index] = item.replace("blocks.", "D")
        index += 1

    short_names = [
        #h.split(".attn.")[0] + "." + h.split(".attn.")[1][:8] if ".attn." in h else h
        h.split(".attn.")[0] + "." + h.split(".attn.")[1] if ".attn." in h else h
        for h in seen_hooks
    ]

    ax.set_xticks(range(len(seen_hooks)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)

    fig.tight_layout()

    outpath = SNAPSHOT_DATA_PATH / f"{filename}.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    #print(f"Chart saved to {outpath}")


def plot_cosine_chart2(filename: str) -> None:
    """Read cosine similarity per-head CSV data and plot a heatmap.

    The CSV is expected at ``snapshots/data/<filename>`` with the format::

        hook1, hook2, head, cosine_similarity

    The heatmap uses hooks as the x-axis and heads as the y-axis, with
    cosine similarity shown as cell color.

    The output image is saved to ``snapshots/data/<filename>.png``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    filepath = SNAPSHOT_DATA_PATH / filename
    if not filepath.is_file():
        print(f"File not found: {filepath}")
        return

    # ── read CSV ──────────────────────────────────────────────────────
    rows: list[tuple[str, int, float]] = []     # (hook, head, cos_sim)
    seen_hooks: list[str] = []
    seen_heads: set[int] = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            hook = parts[0].strip()
            head = int(parts[2].strip())
            cos_sim = float(parts[3].strip())
            rows.append((hook, head, cos_sim))
            if hook not in seen_hooks:
                seen_hooks.append(hook)
            seen_heads.add(head)

    if not rows:
        print("No data rows found.")
        return

    # ── build heatmap matrix ──────────────────────────────────────────
    hook_index_of = {h: i for i, h in enumerate(seen_hooks)}
    n_hooks = len(seen_hooks)
    sorted_heads = sorted(seen_heads)
    n_heads = len(sorted_heads)
    head_index_of = {h: i for i, h in enumerate(sorted_heads)}

    # NaN fill so missing combos show as a distinct color (via cmap.set_bad)
    matrix = np.full((n_heads, n_hooks), np.nan)
    for hook, head, cos_sim in rows:
        matrix[head_index_of[head], hook_index_of[hook]] = cos_sim

    # ── draw heatmap ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_hooks * 0.6), max(4, n_heads * 0.5)))

    cmap = plt.cm.RdYlGn  # red (low) → yellow → green (high)
    cmap.set_bad(color='lightgray')

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", rotation=270, labelpad=15)

    # Tick labels – abbreviated hook names
    index = 0
    for item in seen_hooks:
        seen_hooks[index] = item.replace("blocks.", "D")
        index += 1

    short_names = [
        #h.split(".attn.")[0] + "." + h.split(".attn.")[1][:8] if ".attn." in h else h
        h.split(".attn.")[0] + "." + h.split(".attn.")[1] if ".attn." in h else h
        for h in seen_hooks
    ]
    ax.set_xticks(range(n_hooks))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"Head {h}" for h in sorted_heads], fontsize=8)

    ax.set_xlabel("Hook")
    ax.set_ylabel("Head")
    ax.set_title(f"Cosine similarity heatmap — {filename}")

    # Annotate cells with values if grid is not too large
    if n_hooks * n_heads <= 200:
        for i in range(n_heads):
            for j in range(n_hooks):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="black")

    fig.tight_layout()

    outpath = SNAPSHOT_DATA_PATH / f"{filename}2.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    #print(f"Heatmap saved to {outpath}")


def plot_cosine_chart3(filename: str) -> None:
    """Read cosine similarity per-head CSV data and plot a heatmap.

    The CSV is expected at ``snapshots/data/<filename>`` with the format::

        hook1, hook2, head, cosine_similarity

    The heatmap uses hooks as the x-axis and heads as the y-axis, with
    cosine similarity shown as cell color.

    The output image is saved to ``snapshots/data/<filename>.png``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    filepath = SNAPSHOT_DATA_PATH / filename
    if not filepath.is_file():
        print(f"File not found: {filepath}")
        return

    # ── read CSV ──────────────────────────────────────────────────────
    rows: list[tuple[str, int, float]] = []     # (hook, head, cos_sim)
    seen_hooks: list[str] = []
    seen_heads: set[int] = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            hook = parts[0].strip()
            if "hook_o" not in hook and "hook_z" not in hook:
                continue
            #print(f"hook: {hook}")

            head = int(parts[2].strip())
            cos_sim = float(parts[3].strip())
            rows.append((hook, head, cos_sim))
            if hook not in seen_hooks:
                seen_hooks.append(hook)
            seen_heads.add(head)

    if not rows:
        print("No data rows found.")
        return

    # ── build heatmap matrix ──────────────────────────────────────────
    hook_index_of = {h: i for i, h in enumerate(seen_hooks)}
    n_hooks = len(seen_hooks)
    sorted_heads = sorted(seen_heads)
    n_heads = len(sorted_heads)
    head_index_of = {h: i for i, h in enumerate(sorted_heads)}

    # NaN fill so missing combos show as a distinct color (via cmap.set_bad)
    matrix = np.full((n_heads, n_hooks), np.nan)
    for hook, head, cos_sim in rows:
        matrix[head_index_of[head], hook_index_of[hook]] = cos_sim

    # ── draw heatmap ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_hooks * 0.6), max(4, n_heads * 0.5)))

    cmap = plt.cm.RdYlGn  # red (low) → yellow → green (high)
    cmap.set_bad(color='lightgray')

    im = ax.imshow(matrix, aspect='auto', cmap=cmap, vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity", rotation=270, labelpad=15)

    # Tick labels – abbreviated hook names
    index = 0
    for item in seen_hooks:
        seen_hooks[index] = item.replace("blocks.", "D")
        index += 1

    short_names = [
        h.split(".attn.")[0] + "." + h.split(".attn.")[1] if ".attn." in h else h
        for h in seen_hooks
    ]
    #print(short_names)
    ax.set_xticks(range(n_hooks))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_heads))
    ax.set_yticklabels([f"Head {h}" for h in sorted_heads], fontsize=8)

    ax.set_xlabel("Hook")
    ax.set_ylabel("Head")
    ax.set_title(f"Cosine similarity heatmap — {filename}")

    # Annotate cells with values if grid is not too large
    if n_hooks * n_heads <= 200:
        for i in range(n_heads):
            for j in range(n_hooks):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="black")

    fig.tight_layout()

    outpath = SNAPSHOT_DATA_PATH / f"{filename}_hook_o_z_only.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    #print(f"Heatmap saved to {outpath}")


def plot_cosine_chart4(filename: str) -> None:
    """Read cosine similarity per-head CSV data and plot a line-segment chart.

    The CSV is expected at ``snapshots/data/<filename>`` with the format::

        hook1, hook2, head, cosine_similarity

    For each attention head a piecewise-linear path is drawn.  Each segment
    spans one unit on the x-axis and its *slope* is set to
    ``arccos(cosine_similarity)``.  Cosine similarity close to 1 therefore
    produces a flat segment, while a low similarity produces a steep upward
    segment.

    The output image is saved to ``snapshots/data/<filename>.png``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    filepath = SNAPSHOT_DATA_PATH / filename
    if not filepath.is_file():
        print(f"File not found: {filepath}")
        return

    # ── read CSV ──────────────────────────────────────────────────────
    rows: list[tuple[str, int, float]] = []     # (hook, head, cos_sim)
    seen_hooks: list[str] = []
    seen_heads: set[int] = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) != 4:
                continue
            hook = parts[0].strip()
            head = int(parts[2].strip())
            cos_sim = float(parts[3].strip())
            rows.append((hook, head, cos_sim))
            if hook not in seen_hooks:
                seen_hooks.append(hook)
            seen_heads.add(head)

    if not rows:
        print("No data rows found.")
        return

    hook_index_of = {h: i for i, h in enumerate(seen_hooks)}
    heads: dict[int, list[tuple[int, float]]] = {h: [] for h in seen_heads}
    for hook, head, cos_sim in rows:
        heads[head].append((hook_index_of[hook], cos_sim))

    # ── draw ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    sorted_heads = sorted(seen_heads)
    n_heads = len(sorted_heads)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_heads))
    
    for (head, segments), color in zip(
        sorted(heads.items()), colors
    ):
        x = 0.0
        y = 0.0
        xs: list[float] = [x]
        ys: list[float] = [y]
        for _hook_idx, cos_sim in sorted(segments):
            angle = math.acos(max(-1.0, min(1.0, cos_sim)))  # clamp to [-1, 1]
            x += 1.0
            y = angle
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, color=color, label=f"Head {head}")

    ax.set_xlabel("Hook index (each unit = one hook)")
    ax.set_ylabel("Cumulative arccos(cosine similarity)")
    ax.set_title(f"Cosine similarity change per head — {filename}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

    # Tick labels – abbreviated hook names
    index = 0
    for item in seen_hooks:
        seen_hooks[index] = item.replace("blocks.", "D")
        index += 1

    short_names = [
        #h.split(".attn.")[0] + "." + h.split(".attn.")[1][:8] if ".attn." in h else h
        h.split(".attn.")[0] + "." + h.split(".attn.")[1] if ".attn." in h else h
        for h in seen_hooks
    ]

    ax.set_xticks(range(len(seen_hooks)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)

    fig.tight_layout()

    outpath = SNAPSHOT_DATA_PATH / f"{filename}_4.png"
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    #print(f"Chart saved to {outpath}")


