# Практика 3. Квантование сетей в INT8 формат при помощи инструмента Calibration Tool

## Цели

__Цель данной работы__ - решить задачу калибровки модели в формат INT8, используя инструмент Calibration Tool библиотеки OpenVINO.

## Задачи

__Основные задачи:__

 1. Изучить инструмент Calibration Tool.
 1. Подготовить данные и скрипты для калибровки модели `mobilenet-ssd`.
 1. Калибровать модель `mobilenet-ssd`.
 1. Измерить скорость работы оригинальной модели и модели, калиброванной в INT8.
  
## Общая последовательность действий

 1. Установить зависимости, необходимые для работы Calibration Tool.
 1. Скачать и распаковать набор данных, который будет использован для калибрации.
 1. Разработать скрипты, которыми будет проводиться калибрация сети .
 1. Выполнить калибрацию.
 1. Измерить производительность оригинальной модели и INT8 модели 
 в сихнронном режиме и сравнить их.

## Детальная инструкция по выполнению работы

Данная инструкция основана на оригинальной документации, доступной по [ссылке][calibration-docs].
     
 1. Скачайте тренировочный набор данных Pascal VOC и распакуйте в вашу рабочую директорию. 
            
    Для калибрации модели `mobilenet-ssd` воспользуйтесь датасетом PASCAL-VOC 2007 
    trainval dataset, на котором обучалась данная модель. Поскольку на 
    официальном сайте скачивание не работает, можно воспользоваться 
    [зеркалом][pascal-voc-2007].
 
 1. Активируйте виртуальную среду Python, созданную на [практике 1][practice_1].
    
 1. Перейдите в вашу рабочую папку на компьютере, и установите зависимости для Calibration Tool.
 
    ```bash
    $ cd C:\UNN_HPC_SCHOOL_2019_OPENVINO
    $ python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\open_model_zoo\tools\accuracy_checker\setup.py" install
    ```
 1. Создайте файл формата `.pickle`, который содержит набор изображений и 
 выходов, необходимых для проведения операции калибрации.
 
    Для проведения калибрации достаточно использовать небольшую часть датасета 
    на несколько тысяч или даже сотен изображений, поскольку мы не переобучаем 
    полностью сеть, мы лишь корректируем веса.
    
    ```bash
    $  python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\tools\accuracy_checker_tool\convert_annotation.py" voc_detection --annotations_dir C:\dev\openvino_calibration\VOCdevkit\VOC2007\Annotations --imageset_file C:\dev\openvino_calibration\VOCdevkit\VOC2007\ImageSets\Main\train.txt --images_dir C:\dev\openvino_calibration\VOCdevkit\VOC2007\JPEGImages -a C:\dev\openvino_calibration\train_images.pickle -m C:\dev\openvino_calibration\train_images.meta -ss 500
    ```
    
    Подробное описание можно прочитать по [ссылке][convert_annotation]
 
 1. Создайте файл `sdd_calibrate.yml`, который будет содержать файл эксперимента калибрации:
 
    ```
    ssd_calibrate.yml
    _________________________________________________________________
    models:
      - name: mobilenet-ssd
        launchers:
          - framework: dlsdk
            device: CPU
        model: mobilenet-ssd.xml
            weights: mobilenet-ssd.bin
            adapter: ssd
            cpu_extensions: C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\inference_engine\bin\intel64\Release\cpu_extension_avx2.dll
    
        datasets:
          - name: VOC2007
            data_source: C:\dev\openvino_calibration\VOCdevkit\VOC2007\JPEGImages
            annotation: C:\dev\openvino_calibration\train_images.pickle
            dataset_meta: C:\dev\openvino_calibration\train_images.meta
            preprocessing:
              - type: resize
                size: 300
              - type: normalization
                mean: 104, 117, 123
            postprocessing:
              - type: resize_prediction_boxes
            metrics:
              - type: map
                integral: 11point
                ignore_difficult: True
                presenter: print_scalar
    _________________________________________________________________
    ```
    
    Данный файл содержит в себе описание задачи, которую решает модель, путь до файлов модели, путь до датасета файла `pickle`
    
    Подробное описание можно прочитать по [ссылке][adapters]
    
    
 1. Запустите калибрацию сети
 
    ```bash
    $ python "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\tools\calibration_tool\calibrate.py" \
    -c C:\UNN_HPC_SCHOOL_2019_OPENVINO\config.yml \
    -m C:\UNN_HPC_SCHOOL_2019_OPENVINO\models \
    -s C:\UNN_HPC_SCHOOL_2019_OPENVINO \
    -M "C:\Program Files (x86)\IntelSWTools\openvino_2019.3.379\deployment_tools\model_optimizer" \
    --annotations C:\UNN_HPC_SCHOOL_2019_OPENVINO
    ```
 
 1. Запустите получившуюся INT8 модель в вашем демо-приложении с предыдущей практики.
 
 1. Добавьте в ваше демо-приложение измерение скорости работы модели.
    
    Скорость работы модели как правило измеряют двумя основными метриками:
      - latency - среднее время между одного запуска модели   
      - FPS - количество изображений в секунду, которое может обработать модель

    Среднее время работы модели включает в себя запуск модели с поданной 
    на вход картинкой. В вашем случае это функция ниже: 
    
    ```python
    output = self.exec_net.infer(inputs = {input_blob: blob})
    ```
    
    Для получения более точных данных их необходимо усреднить по большому 
    количеству запусков, например 1000. При этом желательно избаться 
    от значений-выбросов, которые сильно превышают стандартное отклонение.
    
    Для нахождения среднего и среднеквадратичного отклонения воспользуйтесь функциями библиотеки numpy
    
    ```python
    time_std = numpy.std(times)
    time_mean = numpy.mean(times)
    ```    
    
    Для нахождения метрики FPS необходимо разделить все количество обработанных 
    кадров на общее время их обработки. 
    
 1. Сравните скорость работы FP32, FP16 и INT8 варианта модели MobileNet-SSD.
 
<!-- LINKS -->
[calibration-docs]: https://docs.openvinotoolkit.org/2019_R3.1/_inference_engine_tools_calibration_tool_README.html
[pascal-voc-2007]: https://pjreddie.com/projects/pascal-voc-dataset-mirror/
[practice_2]: practice2.md
[adapters]: http://docs.openvinotoolkit.org/2019_R3.1/_tools_accuracy_checker_accuracy_checker_adapters_README.html
[convert_annotation]: http://docs.openvinotoolkit.org/2019_R3.1/_tools_accuracy_checker_accuracy_checker_annotation_converters_README.html