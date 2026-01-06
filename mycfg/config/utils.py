from pathlib import Path
from typing import Any, Dict, Optional, TypeVar, Type, Union
from omegaconf import OmegaConf, DictConfig
from dataclasses import is_dataclass, fields
import typer
import inspect
from functools import wraps
import threading

T = TypeVar('T')

# Thread-local storage for config overrides
_thread_local = threading.local()


def load_config_with_overrides(
    config_class: Type[T],
    config_path: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = None
) -> T:
    """
    Load configuration from YAML file, apply CLI overrides, and return structured config.
    
    Args:
        config_class: Dataclass type to structure the configuration
        config_path: Path to YAML configuration file 
        overrides: Dictionary of CLI overrides to apply
        save_path: Path to save the final configuration (optional)
    
    Returns:
        Instance of config_class with merged configuration
    """
    # Load base configuration from YAML if provided
    if config_path and config_path.exists():
        base_config = OmegaConf.load(config_path)
    else:
        base_config = OmegaConf.create({})
    
    # Create structured config from dataclass
    structured_config = OmegaConf.structured(config_class)
    print(structured_config)
    
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
    
    # Convert back to structured dataclass
    return OmegaConf.to_object(merged_config)


def create_cli_overrides(**kwargs) -> Dict[str, Any]:
    """
    Create overrides dictionary from CLI arguments, filtering out None values.
    
    Args:
        **kwargs: CLI arguments (typically from locals())
        
    Returns:
        Dictionary of non-None overrides, excluding non-config parameters
    """
    # Parameters that are not part of the config schema
    excluded_params = {'config_path', 'save_config'}
    
    return {
        k: v for k, v in kwargs.items() 
        if v is not None and k not in excluded_params
    }


def create_cli_overrides_from_schema(config_class: Type[T], **kwargs) -> Dict[str, Any]:
    """
    Create overrides dictionary from CLI arguments, automatically filtering to only
    include fields that exist in the config schema.
    
    Args:
        config_class: Dataclass type to check fields against
        **kwargs: CLI arguments (typically from locals())
        
    Returns:
        Dictionary of non-None overrides that match config schema fields
    """
    if not is_dataclass(config_class):
        raise ValueError("config_class must be a dataclass")
    
    # Get field names from the dataclass
    config_field_names = {field.name for field in fields(config_class)}
    
    return {
        k: v for k, v in kwargs.items() 
        if v is not None and k in config_field_names
    }


def auto_config_cli(config_class: Type[T]):
    """
    Decorator that automatically adds CLI arguments for all fields in a dataclass
    and makes them available to the function.
    
    Args:
        config_class: Dataclass type to generate CLI arguments from
        
    Returns:
        Decorated function with dynamically added CLI arguments
    """
    def decorator(func):
        if not is_dataclass(config_class):
            raise ValueError("config_class must be a dataclass")
        
        # Get the original function signature
        sig = inspect.signature(func)
        
        # Create new parameters for each config field
        new_params = list(sig.parameters.values())
        
        for field in fields(config_class):
            field_type = field.type
            
            # Handle Optional types and provide appropriate default
            if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                # Already Optional, use as-is
                param_type = field_type
            else:
                # Make it Optional
                param_type = Optional[field_type]
            
            # Create help text from field name
            help_text = f"{field.name.replace('_', ' ').title()}"
            
            # Create the parameter with typer.Option
            param = inspect.Parameter(
                field.name,
                inspect.Parameter.KEYWORD_ONLY,
                default=typer.Option(None, help=help_text),
                annotation=param_type
            )
            new_params.append(param)
        
        # Create new signature
        new_sig = sig.replace(parameters=new_params)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract config field values from kwargs
            config_field_names = {field.name for field in fields(config_class)}
            config_overrides = {k: v for k, v in kwargs.items() if k in config_field_names and v is not None}
            
            # Separate original function params from config params
            sig_params = list(sig.parameters.keys())
            original_kwargs = {k: v for k, v in kwargs.items() if k in sig_params}
            
            # Store config overrides in thread-local storage
            _thread_local.config_overrides = config_overrides
            
            return func(*args, **original_kwargs)
        
        # Apply the new signature
        wrapper.__signature__ = new_sig
        
        return wrapper
    
    return decorator


def get_current_config_overrides():
    """
    Get the current config overrides from the decorated function context.
    This should be called from within a function decorated with @auto_config_cli.
    """
    return getattr(_thread_local, 'config_overrides', {})