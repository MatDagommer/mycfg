import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path
    from typing import Any, Dict, Optional, TypeVar, Type
    from dataclasses import is_dataclass, fields

    from mycfg.config.schemas import TrainingConfig
    return (TrainingConfig,)


@app.cell
def _(TrainingConfig):
    cfg = TrainingConfig()
    return (cfg,)


@app.cell
def _(cfg):
    print(cfg)
    return


@app.function
def print_stuff(**kwargs):
    print(kwargs)


@app.cell
def _(DictConfig, cfg):
    print_stuff(**DictConfig(cfg))
    return


@app.cell
def _(OmegaConf, config_class, overrides, save_path):
    config_path = None
    # Load base configuration from YAML if provided
    if config_path and config_path.exists():
        base_config = OmegaConf.load(config_path)
    else:
        base_config = OmegaConf.create({})

    # Create structured config from dataclass
    structured_config = OmegaConf.structured(config_class)

    # Merge base config with structured config (structured takes precedence for defaults)
    merged_config = OmegaConf.merge(structured_config, base_config)

    # Apply CLI overrides if provided
    if overrides:
        # Filter out None values from overrides
        filtered_overrides = {k: v for k, v in overrides.items() if v is not None}
        if filtered_overrides:
            override_config = OmegaConf.create(filtered_overrides)
            merged_config = OmegaConf.merge(merged_config, override_config)

    # Save the final configuration if requested
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(merged_config, save_path)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
