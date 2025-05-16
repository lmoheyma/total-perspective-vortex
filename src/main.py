import mne
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def main():
    root = tk.Tk()
    root.wm_title("Embedding Raw Plot")

    def _quit():
        root.quit()
        root.destroy()

    button = tk.Button(master=root, text="Quit", command=_quit)
    button.pack(side=tk.TOP)

    raw = mne.io.read_raw_edf('/media/lmoheyma/DEBIAN 12_7/dataset/files/S001/S001R08.edf', preload=True)
    
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
    rename_dict = dict(zip(raw.ch_names, channel_names_64))
    raw.rename_channels(rename_dict)

    raw.set_montage("standard_1005")

    print(raw.ch_names)
    fig=raw.compute_psd().plot(spatial_colors=True, show=False)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP)

    tk.mainloop()

if __name__ == '__main__':
    main()
