from django.core.management.base import BaseCommand
from main.models import Images
from django.conf import settings
import os

def get_all_file_names(directory):
    for dirpath, _, filenames in os.walk(directory):
        for name in filenames:
            yield os.path.relpath(os.path.join(dirpath, name), directory).replace('\\', '/')


class Command(BaseCommand):
    
    def handle(self, *args, **options):
        
        path_list = [name for name in get_all_file_names(settings.BASE_DIR / 'main/static/data')]
        models=[Images(file_path=name) for name in path_list]
        Images.objects.bulk_create(models, ignore_conflicts=True)
        