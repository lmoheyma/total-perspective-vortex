import mne
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import glob

def main():
    root = tk.Tk()
    root.wm_title("EEG Data Viewer")
    root.geometry("1200x800")

    # Variables
    current_raw = None
    current_canvas = None
    
    # Channel names for renaming
    channel_names_64 = [
        'Fp1', 'Fpz', 'Fp2',
        'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
        'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
        'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
        'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
        'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
        'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
        'PO7', 'PO3', 'POz', 'PO4', 'PO8',
        'O1', 'Oz', 'O2', 'Iz', 'TP10', 'T9'
    ]

    def load_edf_file(filepath):
        nonlocal current_raw, current_canvas
        
        try:
            # Clear previous plot
            if current_canvas:
                current_canvas.get_tk_widget().destroy()
                current_canvas = None
            
            # Load new file
            status_label.config(text=f"Loading: {os.path.basename(filepath)}")
            root.update()
            
            current_raw = mne.io.read_raw_edf(filepath, preload=True)
            
            # Rename channels if needed
            if len(current_raw.ch_names) == len(channel_names_64):
                rename_dict = dict(zip(current_raw.ch_names, channel_names_64))
                current_raw.rename_channels(rename_dict)
                current_raw.set_montage("standard_1005")
            
            # Get task type from filename
            task_type = get_task_description(os.path.basename(filepath))
            
            # Create plot based on selected view
            view_type = view_combo.get()
            if view_type == "PSD (Power Spectral Density)":
                fig = current_raw.compute_psd().plot(spatial_colors=True, show=False)
            elif view_type == "Raw Signal":
                fig = current_raw.plot(duration=10, scalings='auto', show=False)
            elif view_type == "Frequency Bands":
                fig = plot_frequency_bands(current_raw)
            else:
                fig = current_raw.compute_psd().plot(spatial_colors=True, show=False)
            
            # Embed plot in tkinter
            current_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
            current_canvas.draw()
            current_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Update status
            status_label.config(text=f"Loaded: {os.path.basename(filepath)} - {len(current_raw.ch_names)} channels")
            file_info_label.config(text=f"Task: {task_type} | Duration: {current_raw.times[-1]:.1f}s | Sampling rate: {current_raw.info['sfreq']:.1f}Hz")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load EDF file: {e}")
            status_label.config(text="Error loading file")

    def get_task_description(filename):
        """Get task description from EDF filename"""
        # Extract run number from filename (e.g., S001R08.edf -> R08)
        if 'R' in filename:
            run_num = filename.split('R')[1].split('.')[0]
            run_int = int(run_num)
            
            if run_int == 1:
                return "Baseline - Eyes Open"
            elif run_int == 2:
                return "Baseline - Eyes Closed"
            elif run_int in [3, 7, 11]:
                return "Motor Execution - Left/Right Fist"
            elif run_int in [4, 8, 12]:
                return "Motor Imagery - Left/Right Fist"
            elif run_int in [5, 9, 13]:
                return "Motor Execution - Both Fists/Feet"
            elif run_int in [6, 10, 14]:
                return "Motor Imagery - Both Fists/Feet"
            else:
                return f"Run {run_num}"
        return "Unknown Task"

    def plot_frequency_bands(raw):
        """Create frequency band analysis plot"""
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Define frequency bands
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
        
        # Compute PSD
        psd, freqs = raw.compute_psd(fmax=50).get_data(return_freqs=True)
        
        # Calculate power in each band
        band_powers = {}
        for band_name, (fmin, fmax) in bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            band_powers[band_name] = np.mean(psd[:, freq_mask], axis=1)
        
        # Create plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot each frequency band
        for i, (band_name, power) in enumerate(band_powers.items()):
            if i < len(axes):
                im = axes[i].scatter(range(len(power)), power, c=power, cmap='viridis', s=50)
                axes[i].set_title(f'{band_name} Band Power')
                axes[i].set_xlabel('Channel')
                axes[i].set_ylabel('Power (dB)')
                plt.colorbar(im, ax=axes[i])
        
        # Hide unused subplot
        if len(axes) > len(band_powers):
            axes[-1].set_visible(False)
        
        plt.tight_layout()
        return fig

    def on_view_change(event):
        """Reload current file with new view"""
        if current_raw is not None:
            selected_patient = patient_combo.get()
            selected_test = test_combo.get()
            
            if (selected_patient and selected_patient != "Select a patient..." and 
                selected_test and selected_test not in ["Select a test...", "No EDF files found", "Error loading tests"]):
                
                filepath = os.path.join(base_directory, selected_patient, selected_test)
                load_edf_file(filepath)

    def browse_directory():
        directory = filedialog.askdirectory(
            title="Select directory containing patient folders",
            initialdir="/mnt/c/Users/louis/Downloads/archive/files/"
        )
        if directory:
            global base_directory
            base_directory = directory
            update_patient_list()

    def update_patient_list():
        patient_combo['values'] = []
        test_combo['values'] = []
        try:
            # Find all patient directories (S001, S002, etc.)
            patient_dirs = []
            for item in os.listdir(base_directory):
                item_path = os.path.join(base_directory, item)
                if os.path.isdir(item_path) and item.startswith('S'):
                    patient_dirs.append(item)
            
            patient_dirs.sort()
            patient_combo['values'] = patient_dirs
            
            if patient_dirs:
                patient_combo.set("Select a patient...")
                
        except Exception as e:
            print(f"Error updating patient list: {e}")

    def on_patient_select(event):
        selected_patient = patient_combo.get()
        if selected_patient and selected_patient != "Select a patient...":
            update_test_list(selected_patient)

    def update_test_list(patient):
        test_combo['values'] = []
        try:
            patient_dir = os.path.join(base_directory, patient)
            edf_files = []
            
            # Find all EDF files in the patient directory
            for file in os.listdir(patient_dir):
                if file.endswith('.edf'):
                    edf_files.append(file)
            
            edf_files.sort()
            test_combo['values'] = edf_files
            
            if edf_files:
                test_combo.set("Select a test...")
            else:
                test_combo.set("No EDF files found")
                
        except Exception as e:
            print(f"Error updating test list: {e}")
            test_combo.set("Error loading tests")

    def on_test_select(event):
        selected_patient = patient_combo.get()
        selected_test = test_combo.get()
        
        if (selected_patient and selected_patient != "Select a patient..." and 
            selected_test and selected_test not in ["Select a test...", "No EDF files found", "Error loading tests"]):
            
            filepath = os.path.join(base_directory, selected_patient, selected_test)
            load_edf_file(filepath)

    def _quit():
        root.quit()
        root.destroy()

    # Create main frame
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Control panel
    control_frame = tk.Frame(main_frame)
    control_frame.pack(fill=tk.X, pady=(0, 5))

    tk.Button(control_frame, text="Browse Directory", command=browse_directory).pack(side=tk.LEFT, padx=(0, 5))
    tk.Button(control_frame, text="Quit", command=_quit).pack(side=tk.RIGHT)

    # Selection frame
    selection_frame = tk.Frame(main_frame)
    selection_frame.pack(fill=tk.X, pady=(0, 5))

    # Patient selection
    tk.Label(selection_frame, text="Patient:").pack(side=tk.LEFT, padx=(0, 5))
    patient_combo = ttk.Combobox(selection_frame, width=15, state="readonly")
    patient_combo.pack(side=tk.LEFT, padx=(0, 10))
    patient_combo.bind('<<ComboboxSelected>>', on_patient_select)

    # Test selection
    tk.Label(selection_frame, text="Test:").pack(side=tk.LEFT, padx=(0, 5))
    test_combo = ttk.Combobox(selection_frame, width=20, state="readonly")
    test_combo.pack(side=tk.LEFT, padx=(0, 10))
    test_combo.bind('<<ComboboxSelected>>', on_test_select)

    # View selection
    tk.Label(selection_frame, text="View:").pack(side=tk.LEFT, padx=(0, 5))
    view_combo = ttk.Combobox(selection_frame, width=25, state="readonly")
    view_combo['values'] = ["PSD (Power Spectral Density)", "Raw Signal", "Frequency Bands"]
    view_combo.set("PSD (Power Spectral Density)")
    view_combo.pack(side=tk.LEFT, padx=(0, 10))
    view_combo.bind('<<ComboboxSelected>>', on_view_change)

    # Content frame with plot
    content_frame = tk.Frame(main_frame)
    content_frame.pack(fill=tk.BOTH, expand=True)

    # Plot frame
    plot_frame = tk.Frame(content_frame)
    plot_frame.pack(fill=tk.BOTH, expand=True)

    # Status bar
    status_frame = tk.Frame(main_frame)
    status_frame.pack(fill=tk.X, pady=(5, 0))

    status_label = tk.Label(status_frame, text="Ready - Select an EDF file to begin", relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(fill=tk.X)

    file_info_label = tk.Label(status_frame, text="", relief=tk.SUNKEN, anchor=tk.W)
    file_info_label.pack(fill=tk.X)

    # Load default directory if it exists
    default_dir = "/mnt/c/Users/louis/Downloads/archive/files/"
    if os.path.exists(default_dir):
        base_directory = default_dir
        update_patient_list()

    tk.mainloop()

if __name__ == '__main__':
    main()
