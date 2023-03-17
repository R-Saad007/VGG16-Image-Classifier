from libraries import *
from normalize_load_data import *
from vgg import VGG16

IMG_PATH = './inference_images/images/'
MODEL_PATH = './cifar_net.pth'
# inference function
def inference():
    # list to store predicted labels
    pred_labels = []
    # list of actual labels
    ground_truth_labels = classes
    # To load saved model
    VGG = VGG16()
    VGG.load_state_dict(torch.load(MODEL_PATH))
    # to set dropout and batch normalization layers to evaluation 
    VGG.eval()
    VGG = VGG.to(device)
    # transformation on images
    data_transforms = transforms.Compose([
    # conversion to tensors
    transforms.ToTensor(),
    # image resize
    transforms.Resize((32,32)),
    # normalizing tensors
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ((mean of 0.5 for R,G,B) (sd of 0.5 for R,G,B))
    ])
    # inference dataset
    inference_dataset = torchvision.datasets.ImageFolder('./inference_images', transform=data_transforms)
    dataloader = torch.utils.data.DataLoader(inference_dataset, batch_size = 1, shuffle=False, num_workers=NUM_OF_WORKERS)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            output = VGG(inputs)
            output = output.to(device)
            _, predicted = torch.max(output.data, 1)
            pred_labels.append(predicted)
    # printing class assigned to assess model inference
    for x in range(0, len(pred_labels)):
        print(f"Class Assigned to image{x+1}:", ground_truth_labels[pred_labels[x]])
    return

# function to display inference images given by the user
def view_images():
    img_list = sorted(os.listdir(IMG_PATH))
    # list to store all images
    images = []
    for image in img_list:
        image_path = os.path.join(IMG_PATH, image)
        images.append(cv.imread(image_path))
           
    fig = plt.figure(figsize=(8, 8))
    rows = 5
    for x in range(len(images)):
        fig.add_subplot(rows, 5, x+1)
        plt.title(f'{x+1}')
        plt.imshow(images[x])
        plt.axis('off')
    plt.savefig('inference_input.png')
    return

if __name__ == '__main__':
    # For calculating execution time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    print("Starting Image Inference")
    # whatever you are timing goes here
    inference()
    view_images()
    end.record()
    # Waits for everything to finish running
    torch.cuda.synchronize()
    print("Execution Time:","%.3f" % (start.elapsed_time(end)/1000), "seconds")  # seconds
