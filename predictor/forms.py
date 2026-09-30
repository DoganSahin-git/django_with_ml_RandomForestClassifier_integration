from django import forms

_field = lambda label: forms.FloatField(label=label, min_value=0, max_value=20)


class IrisForm(forms.Form):
    sepal_length = _field("Sepal length (cm)")
    sepal_width = _field("Sepal width (cm)")
    petal_length = _field("Petal length (cm)")
    petal_width = _field("Petal width (cm)")

    def as_features(self):
        d = self.cleaned_data
        return [d["sepal_length"], d["sepal_width"], d["petal_length"], d["petal_width"]]
