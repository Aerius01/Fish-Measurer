"""Modern, validated numeric entry widget for the Fish Measurer application."""

import tkinter as tk
from typing import Optional, Callable, Union, Any
from abc import ABC, abstractmethod


class ValueValidator(ABC):
    """Abstract base class for value validators."""
    
    @abstractmethod
    def validate(self, value: str) -> str:
        """Validate and return corrected value."""
        pass


class NumericValidator(ValueValidator):
    """Validator for numeric values with optional bounds."""
    
    def __init__(self, 
                 lower: Optional[float] = None, 
                 upper: Optional[float] = None,
                 allow_decimal: bool = True):
        self.lower = lower
        self.upper = upper
        self.allow_decimal = allow_decimal
    
    def validate(self, value: str) -> str:
        """Validate numeric input and apply bounds."""
        if not value:
            return value
        
        # Keep only digits and optionally decimal point
        if self.allow_decimal:
            cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
            # Ensure only one decimal point
            if cleaned.count('.') > 1:
                parts = cleaned.split('.')
                cleaned = parts[0] + '.' + ''.join(parts[1:])
        else:
            cleaned = ''.join(c for c in value if c.isdigit())
        
        if cleaned and cleaned != '.':
            try:
                num_value = float(cleaned)
                
                # Apply bounds
                if self.lower is not None and num_value < self.lower:
                    return str(self.lower)
                if self.upper is not None and num_value > self.upper:
                    return str(self.upper)
                
                return cleaned
            except ValueError:
                return ''.join(c for c in cleaned if c.isdigit())
        
        return cleaned


class ValidatedEntry(tk.Frame):
    """A modern entry widget with validation and change callbacks."""
    
    def __init__(self, 
                 parent: tk.Widget,
                 label_text: str,
                 default_value: Union[str, int, float] = "",
                 validator: Optional[ValueValidator] = None,
                 on_change: Optional[Callable[[str], None]] = None,
                 row: int = 0,
                 **kwargs):
        
        super().__init__(parent, **kwargs)
        
        self.validator = validator
        self.on_change = on_change
        self._setting_value = False  # Prevent recursion during validation
        
        # Configure grid
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        
        # Create label
        self.label = tk.Label(self, text=label_text, pady=3)
        self.label.grid(row=0, column=0, sticky='w', padx=5)
        
        # Create StringVar with validation
        self.var = tk.StringVar(self)
        self.var.trace_add('write', self._on_value_changed)
        
        # Create entry
        self.entry = tk.Entry(self, textvariable=self.var, justify='center')
        self.entry.grid(row=0, column=1, sticky='ew', padx=5)
        
        # Set default value
        self.set_value(str(default_value))
        
        # Grid this widget in parent
        self.grid(row=row, column=0, columnspan=2, sticky='ew', pady=2)
    
    def _on_value_changed(self, *args) -> None:
        """Handle value changes with validation."""
        if self._setting_value:
            return
        
        current_value = self.var.get()
        
        # Apply validation
        if self.validator:
            validated_value = self.validator.validate(current_value)
            if validated_value != current_value:
                self._setting_value = True
                self.var.set(validated_value)
                self._setting_value = False
                current_value = validated_value
        
        # Call change callback
        if self.on_change and not self._setting_value:
            self.on_change(current_value)
    
    def get_value(self) -> str:
        """Get current value."""
        return self.var.get()
    
    def set_value(self, value: Union[str, int, float]) -> None:
        """Set value programmatically."""
        self._setting_value = True
        self.var.set(str(value))
        self._setting_value = False
    
    def get_numeric_value(self) -> Optional[float]:
        """Get value as float, return None if invalid."""
        try:
            value = self.get_value()
            return float(value) if value else None
        except ValueError:
            return None
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the entry."""
        state = "normal" if enabled else "disabled"
        self.entry.configure(state=state)
    
    def focus(self) -> None:
        """Focus the entry widget."""
        self.entry.focus_set()


# Factory functions for common use cases
def create_exposure_entry(parent: tk.Widget, 
                         on_change: Optional[Callable[[str], None]] = None,
                         row: int = 0) -> ValidatedEntry:
    """Create an exposure time entry (minimum 42ms)."""
    validator = NumericValidator(lower=42.0, allow_decimal=True)
    return ValidatedEntry(
        parent=parent,
        label_text="Exposure (ms):",
        default_value=2000,
        validator=validator,
        on_change=on_change,
        row=row
    )


def create_framerate_entry(parent: tk.Widget,
                          on_change: Optional[Callable[[str], None]] = None,
                          row: int = 0) -> ValidatedEntry:
    """Create a framerate entry."""
    validator = NumericValidator(lower=1.0, upper=60.0, allow_decimal=True)
    return ValidatedEntry(
        parent=parent,
        label_text="Framerate (fps):",
        default_value=30,
        validator=validator,
        on_change=on_change,
        row=row
    )


def create_frame_count_entry(parent: tk.Widget,
                            on_change: Optional[Callable[[str], None]] = None,
                            row: int = 0) -> ValidatedEntry:
    """Create a frame count entry (minimum 3)."""
    validator = NumericValidator(lower=3, allow_decimal=False)
    return ValidatedEntry(
        parent=parent,
        label_text="Number of Frames:",
        default_value=10,
        validator=validator,
        on_change=on_change,
        row=row
    )
            