from django.shortcuts import render
import torch
import clip
from PIL import Image
import time
from django.http import JsonResponse
from django.http import HttpResponse
from main.models import Images
import requests
from io import BytesIO
import numpy as np
from pgvector.django import L2Distance
import functools
from functools import wraps
import time

# Create your views here.

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def index(request):
    return render(request, 'main/index.html')


def is_image(file):
    try:
        img = Image.open(file)
        return img.format in ['JPEG', 'JPG', 'PNG', 'GIF']
    except:
        return False


def search(request):
    query = request.POST.get('query', '')
    print(query)
    text_embedding = text_to_embedding(query)
    print(text_embedding)
    images = Images.objects.order_by(L2Distance('embedding', text_embedding))[:5]
    return render(request, 'main/search.html', {
        'images': images,
    })


@timeit 
@functools.cache
def text_to_embedding(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model, preprocess = clip.load("ViT-L/14", device=device)
    text = clip.tokenize(text).to(device)
    text_features = model.encode_text(text)
    return text_features.cpu().detach().numpy().astype(np.float32).flatten()
    

def process_image(request):
    
    all_images = Images.objects.filter(embedding__isnull=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model, preprocess = clip.load("ViT-L/14", device=device)
    
    comment = request.POST.get('comment', '')
    
    text = clip.tokenize("a diagram").to(device)
    
    text_features = model.encode_text(text)
# Цикл по всем объектам
    start_time = time.time()
    for image in all_images:
        id_value = image.id
        file_path_value = image.file_path
        
        response = requests.get(f"https://ik.imagekit.io/zwymr4sxm/archive/data/{file_path_value}")
        if response.status_code == 200:
            image_preprocess = preprocess(Image.open(BytesIO(response.content))).unsqueeze(0).to(device)
            
            with torch.no_grad():
                image_features = model.encode_image(image_preprocess)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy().astype(np.float32)
            image.embedding = image_features.flatten()
            image.save()
        else:
            print("Ошибка при загрузке изображения")
        
        # image_preprocess = preprocess(Image.open(f"https://ik.imagekit.io/zwymr4sxm/archive/data/{file_path_value}")).unsqueeze(0).to(device)

        print(f"ID: {id_value}, File Path: {file_path_value}")
    execution_time = time.time() - start_time
    return render(request, 'main/index.html', {
        'results': text_features,
        'execution_time': execution_time
    })
    
    # if request.method == 'POST':
    #     uploaded_file = request.FILES['myfile']
    #     comment = request.POST.get('comment', '')

    #     if is_image(uploaded_file):
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         model, preprocess = clip.load("ViT-B/32", device=device)

    #         image = preprocess(Image.open(uploaded_file)).unsqueeze(0).to(device)
    #         list1 = ["an awp Dragon Lore","an awp Atheris", "an awp Asimov", "an awp Fade", "an awp Worm God"]
    #         text = clip.tokenize(list1).to(device)

    #         start_time = time.time()

    #         with torch.no_grad():
    #             image_features = model.encode_image(image)
    #             text_features = model.encode_text(text)

    #             logits_per_image, logits_per_text = model(image, text)
    #             probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    #         # torch.cuda.synchronize()

    #         execution_time = time.time() - start_time

    #         probs_str = str(probs)
    #         elements = probs_str.strip('[]').split()

    #         # Преобразовать каждый элемент в число
    #         probs_list = [float(element) for element in elements]

    #         # Результаты обработки
    #         results = list(zip(list1, probs_list))

    #         return render(request, 'main/index.html', {
    #             'results': results,
    #             'execution_time': execution_time
    #         })
    #     else:
    #         return HttpResponse('Загруженный файл не является изображением.')

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


