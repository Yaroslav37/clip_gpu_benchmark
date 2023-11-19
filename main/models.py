from django.db import models

# Create your models here.


from pgvector.django import VectorField

class Images(models.Model):
    id = models.AutoField(primary_key=True)
    file_path = models.TextField(null=False, unique=True)
    embedding = VectorField(dimensions=768, null=True)
    
