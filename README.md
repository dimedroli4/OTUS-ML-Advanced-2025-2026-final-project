# Датасет

https://huggingface.co/datasets/netop/gotsf-ds

# Постановка задачи

Будем пытаться предсказать количество пользователей (активность) на луче (beam) на неделю (168 часов) вперед. 

Это ровно доступная нам тестовая выборка (MR_number_test_5w-6w.csv).

Выборки довольно большие, по-этому я не буду выкладывать их в git репозиторий, но их в любой момент можно скачать по ссылке выше. 

# Материалы

- [EDA.ipynb](EDA.ipynb) блокнот с exploratory data analysis. 
- [baseline.ipynb](baseline.ipynb) baseline модель с PoC (proof of concept) что гипотеза о предсказанием вообще состоятельна.
- [XGBoost+LinearRegression+Seasonal.ipynb](XGBoost%2BLinearRegression%2BSeasonal.ipynb) блокнот с улучшениями baseline модели при помощи применения других подходов.
- [final-pipeline.py](final-pipeline.py) Финальный e2e pipeline, который обучает и сохраняет ансамблевую модель.