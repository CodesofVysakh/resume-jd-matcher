from django.urls import path

from .views import analyze_resume_view, health_check

urlpatterns = [
     path("health/", health_check, name="health-check"),
    path("analyze/", analyze_resume_view, name="analyze-resume"),
]