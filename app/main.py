"""Application for electricity demand forecasting"""

import os
import threading
import datetime
from tkinter import Tk, filedialog, Canvas, PhotoImage
from tkinter.messagebox import showinfo
from tkinter.ttk import Style, Button
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from pandas.api.types import is_numeric_dtype


def predict(
        filename: str = 'data.csv',
        target: str = 'Потребление, МВт*ч',
        analysis_length: int = 72,
        preemption_length: int = 168,
        callback=None
):
    energo = pd.read_csv(filename)
    labels = [i for i in energo.columns if is_numeric_dtype(energo[i]) and i != target]

    if target not in energo.columns or not is_numeric_dtype(energo[target]):
        return

    names = labels + [target]
    energo = energo[names]

    scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1)).fit(energo)

    n_old = energo.shape[0]
    train_x = [scaler.transform(energo)[i: i + analysis_length] for i in range(n_old - analysis_length)]

    sh = np.array(train_x).shape[2]
    s_train_x = [[] for _ in range(sh)]

    for i in train_x:
        for j in range(sh):
            s_train_x[j].append(i.T[j])

    gr_train_x = list(map(np.concatenate, train_x))
    train_subs = np.array(scaler.transform(energo)[analysis_length - n_old:]).T
    train_y = train_subs[-1]

    sub_models = [MLPRegressor(hidden_layer_sizes=(500,), max_iter=1000, random_state=1) for _ in train_subs[:-1]]
    g_model = MLPRegressor(hidden_layer_sizes=(500,), max_iter=1000, random_state=1)
    _ = g_model.fit(gr_train_x, train_y)

    for i in range(len(sub_models)):
        sub_models[i].fit(s_train_x[i], train_subs[i])

    series = scaler.transform(energo)[-analysis_length:].T.tolist()

    for _ in range(preemption_length):
        for i in range(sh - 1):
            series[i].append(sub_models[i].predict([series[i][-analysis_length:]])[0])

    for _ in range(preemption_length):
        s = np.array([i[-analysis_length:] for i in series]).reshape(-1)
        series[-1].append(g_model.predict([s])[0])

    series = scaler.inverse_transform(np.array(series).T).T

    out_data = series[-1][analysis_length - 1:]
    out_name = f'out/{datetime.datetime.now()}'.replace(':', '')

    pd.DataFrame({target: out_data}).to_csv(f'{out_name}.csv')

    plt.figure(figsize=(8, 4))
    plt.plot(out_data, color='#630DA7')
    plt.ylabel(target)
    plt.savefig(f'{out_name}.png')

    if callback:
        callback(out_name)


class App(Tk):
    """Toplevel widget of Tk"""
    def __init__(self):
        """Initialization of window"""
        super().__init__()
        if not os.path.isdir("out"):
            os.mkdir("out")

        PADY = 6
        PADX = 8

        self.filename = None

        self.title("Прогнозирование электрических нагрузок")
        # self.geometry("512x512")
        self.config(bg="#544B64", cursor="arrow")
        self.img = None

        try:
            self.iconbitmap("icon.ico")
        except Exception as e:
            print(e)

        stl = Style()
        stl.configure('Button.TLabel', padding=6, anchor="center",
                      foreground='white', background='#544B64', relief='flat')
        stl.map('Button.TLabel', background=[
            ('!pressed', '!active', '#630DA7'),
            ('pressed', '#530097'),
            ('active', '#731DB7')
        ])

        for c in range(2):
            self.columnconfigure(index=c, weight=1)
        for r in range(3):
            self.rowconfigure(index=r, weight=1)

        browse_btn = Button(command=self.browse_files, text="Выбрать файл",
                            style='Button.TLabel', cursor='hand2')
        browse_btn.grid(row=0, column=0, columnspan=1,
                        pady=PADY, padx=PADX)

        self.predict_btn = Button(command=self.predict, text="Сделать прогноз",
                                  style='Button.TLabel', cursor='hand2')
        self.predict_btn.grid(row=0, column=1, columnspan=1,
                              pady=PADY, padx=PADX)
        self.predict_btn["state"] = "disable"

        self.canvas = Canvas(height=400, width=800)
        self.canvas.grid(row=1, column=0, columnspan=2, rowspan=2,
                         pady=PADY, padx=PADX)

    def browse_files(self):
        self.filename = filedialog.askopenfilename(
            initialdir=".", title="Select a File",
            filetypes=(("Text files", "*.csv*"), ("all files", "*.*")))

        if self.filename:
            self.predict_btn["state"] = "enable"
        else:
            self.predict_btn["state"] = "disable"

    def predict(self):
        self.config(cursor="watch")
        threading.Thread(target=predict, kwargs={
            "filename": self.filename,
            "callback": self.callback,
        }).start()

    def callback(self, name, *_):
        self.config(cursor="arrow")

        self.img = PhotoImage(file=f"{name}.png")
        self.canvas.create_image(0, 0, anchor='nw', image=self.img)

        return showinfo("Success", f"{name} сохранено")


if __name__ == '__main__':
    app = App()
    app.mainloop()
