import cv2
import torch
from lenet import LeNet5
from torchvision import transforms
from PIL import Image
import numpy as np

device = torch.device('cpu')
transform = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor()
    ])


def predict(curr_frame):
    net = LeNet5()
    net.load_state_dict(torch.load('params.pkl'))
    net.to(device)
    torch.no_grad()

    trans_frame = transform(curr_frame).unsqueeze(0)

    input_frame = trans_frame.to(device)

    outputs = net(input_frame)
    _, predicted = torch.max(outputs, 1)

#   print("predict number is:", outputs)
    print("number is:", predicted)


# 0 is camera 
def get_image_from_camera():
    capture = cv2.VideoCapture(0)
    i = 0
    while 1:
        ret, frame = capture.read()
        window_name = "digital recognition"
        cv2.imshow(window_name, frame)

        image_resize = cv2.resize(frame, (28, 28), interpolation=cv2.INTER_CUBIC)

        image_gray = cv2.cvtColor(image_resize, cv2.COLOR_RGB2GRAY)
        
        for i in range(0,28):
            for j in range(0, 28):
                curr_pixel = image_gray[i,j]
                image_gray[i,j] = 255 - curr_pixel
        cv2.imshow("convert", image_gray)
        pit_frame = Image.fromarray(image_gray)
        
        predict(pit_frame) 

        if cv2.waitKey(100) & 0xff == ord('q'):
            break;

    capture.release()
    cv2.destroyAllWindows()

def main():
    get_image_from_camera()

if __name__ == '__main__':
    main()
