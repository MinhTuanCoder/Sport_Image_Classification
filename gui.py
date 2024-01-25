import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np

from tensorflow.keras.preprocessing import image


from keras.models import load_model
model = load_model('ModelVGG16_Classification.hdf5')

classes = { 
    0:'Cầu lông',
    1:'Bóng chày',
    2:'Karate',
    3:'Bóng đá',
    4:'Bơi lội',
    5:'Quần vợt',
    6:'Đấu vật',
}
#initialise GUI
def classify(file_path):
    img_path=file_path
    img = image.load_img(img_path, target_size=(224, 224))
    img=np.array(img)/255
    img_batch = np.expand_dims(img, axis=0)
    predict=classes[np.argmax(model.predict(img_batch))]
    label.configure(foreground='#011638', text=predict) 
    

def show_classify_button(file_path):
    btn_classify = Button(top, text="Phân loại", command=lambda: classify(file_path), padx=10, pady=5)
    btn_classify.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    btn_classify.place(relx=0.3,rely=0.85)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text="")
        show_classify_button(file_path)
    except:
        pass
    
    
top=tk.Tk()
top.geometry('800x600')
top.title('Chương trình nhận diện môn thể thao')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD',foreground="#FF0000",font=('arial',15,'bold'))
sign_image = Label(top)
btn_upload=Button(top,text="Chọn Ảnh",command=upload_image,padx=10,pady=5)
btn_upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
btn_upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Hãy chọn một ảnh để phân loại",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()