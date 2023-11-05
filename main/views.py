from django.shortcuts import render
import torch
import clip
from PIL import Image
import time
from django.http import JsonResponse
from django.http import HttpResponse

# Create your views here.

def index(request):
    return render(request, 'main/index.html')


def is_image(file):
    try:
        img = Image.open(file)
        return img.format in ['JPEG', 'JPG', 'PNG', 'GIF']
    except:
        return False

def process_image(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['myfile']
        comment = request.POST.get('comment', '')

        if is_image(uploaded_file):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)

            image = preprocess(Image.open(uploaded_file)).unsqueeze(0).to(device)
            list1 = ["an awp Dragon Lore","an awp Atheris", "an awp Asimov", "an awp Fade", "an awp Worm God"]
            text = clip.tokenize(list1).to(device)

            start_time = time.time()

            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()

            torch.cuda.synchronize()

            execution_time = time.time() - start_time

            probs_str = str(probs)
            elements = probs_str.strip('[]').split()

            # Преобразовать каждый элемент в число
            probs_list = [float(element) for element in elements]

            # Результаты обработки
            results = list(zip(list1, probs_list))

            return render(request, 'main/index.html', {
                'results': results,
                'execution_time': execution_time
            })
        else:
            return HttpResponse('Загруженный файл не является изображением.')

    return render(request, 'main/index.html')


# def process_image(request):
#     result = None

#     if request.method == 'POST':
#         form = ImageUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             image = form.cleaned_data['image']
#             torch.cuda.synchronize()

#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             model, preprocess = clip.load("ViT-B/32", device=device)

#             image = preprocess(Image.open(image)).unsqueeze(0).to(device)
#             list = ["an awp Dragon Lore","an awp Atheris", "an awp Asimov", "an awp Fade", "an awp Worm God"]
#             text = clip.tokenize(list).to(device)

#             start_time = time.time()

#             with torch.no_grad():
#                 image_features = model.encode_image(image)
#                 text_features = model.encode_text(text)
            
#                 logits_per_image, logits_per_text = model(image, text)
#                 probs = logits_per_image.softmax(dim=-1).cpu().numpy()

#             torch.cuda.synchronize()

#             execution_time = time.time() - start_time

#             probs_str = str(probs)
#             elements = probs_str.strip('[]').split()

#             # Преобразовать каждый элемент в число
#             probs_list = [float(element) for element in elements]

#             results = []
#             for i in range(len(probs_list)):
#                 results.append((list[i], probs_list[i]))

#     else:
#         form = ImageUploadForm()

#     return render(request, 'process_image.html', {'form': form, 'results': results})


