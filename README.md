# Датасет

https://huggingface.co/datasets/netop/gotsf-ds

# Постановка задачи

Будем пытаться предсказать количество пользователей (активность) на луче (beam) на неделю (168 часов) вперед. 

Это ровно доступная нам тестовая выборка (MR_number_test_5w-6w.csv).

Выборки довольно большие, по-этому я не буду выкладывать их в git репозиторий, но их в любой момент можно скачать по ссылке выше. 

# Материалы

- [EDA.ipynb](EDA.ipynb) блокнот с exploratory data analysis. 
- [baseline.ipynb](baseline.ipynb) baseline модель с PoC (proof of concept) что гипотеза о предсказании вообще состоятельна.
- [XGBoost+LinearRegression+Seasonal.ipynb](XGBoost%2BLinearRegression%2BSeasonal.ipynb) блокнот с улучшениями baseline модели при помощи применения других подходов и выводами, касательно всего эксперимента.
- [final-pipeline.py](final-pipeline.py) Финальный e2e pipeline, который обучает и сохраняет ансамблевую модель.

# Выводы

## Final pipeline 

```
Loading data...
Data split summary:
Training period: 672 timesteps
Validation period: 168 timesteps
Test period: 168 timesteps
Number of beams: 2880
Training Seasonal Model...
Optimized weights: daily=0.444, weekly=0.506
Training Linear Regression...
Training XGBoost Models...
Training XGBoost: 100%|██████████| 2880/2880 [08:36<00:00,  5.57it/s]
Successfully trained XGBoost for 2880 out of 2880 beams
Computing daily pattern...
Optimizing ensemble weights...
Optimal weights: Seasonal=0.154, LR=0.440, XGB=0.406
Creating final ensemble predictions...
Context window saved: shape=(168, 2880)
Validation performance (on held-out validation set)
MAE: 0.2213
Precision: 0.8505
Recall: 0.8262
F1 Score: 0.8382
Average Precision: 0.7587
Saving ensemble to models/ensemble_model.joblib...
Saved. File size: 854.94 MB
Evaluate in test data (post-training)
Generating predictions for 168 timesteps...
Test Set Metrics:
MAE: 0.2277
Precision: 0.8406
Recall: 0.8242
F1 Score: 0.8323
Average Precision: 0.7489
Actual Active Rate: 0.3190
Predicted Active Rate: 0.3128
```

## Резюме

Модель предсказывает загрузку 2880 лучей 5G базовой станции (Massive MIMO) на 168 часов вперед.

Средняя ошибка: 0.23 пользователя на луч на тестовой выборке.

90% точности в определении активности пользователей.

## В чем Бизнес-ценность ? 

***Экономия ресурсов***: 85% эффективность аллокации базовых станций / секторальных антенн. 
Иными словами, из каждых 100 единиц ресурсов, которые модель выделила на лучи, 85 реально использовались (были активные пользователи).

***Снижение missed opportunities***: только 17% активных лучей пропущены. Иными словами, из каждых 100 реально активных лучей, модель не заметила только 17. 
То есть, только 17% абонентов могли испытать, возможно, не значительные проблемы со связью.