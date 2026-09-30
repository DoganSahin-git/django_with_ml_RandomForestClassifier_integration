import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import ml
from .forms import IrisForm


def index(request):
    result = None
    if request.method == "POST":
        form = IrisForm(request.POST)
        if form.is_valid():
            result = ml.predict(form.as_features())
    else:
        form = IrisForm()
    return render(request, "predictor/index.html", {"form": form, "result": result})


@csrf_exempt  # demo only; use proper auth/tokens in production
@require_POST
def api_predict(request):
    try:
        payload = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    form = IrisForm(payload)
    if not form.is_valid():
        return JsonResponse({"errors": form.errors}, status=400)
    return JsonResponse(ml.predict(form.as_features()))
