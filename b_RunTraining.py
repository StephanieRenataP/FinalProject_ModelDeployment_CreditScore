"""
================================================================================
 Entry point CLI untuk menjalankan training pipeline.
================================================================================
File ini SENGAJA dipisah dari `training_pipeline.py`.

Alasan: ketika sebuah file Python dijalankan langsung (`python file.py`),
Python mengikat seluruh kelas yang didefinisikan di file tersebut ke modul
`__main__`. Jika `CreditDataCleaner` ikut terikat ke `__main__`, maka artefak
`.pkl` yang dihasilkan hanya bisa di-load ulang oleh skrip yang juga
menjalankan `training_pipeline.py` sebagai `__main__` — TIDAK bisa di-load
oleh skrip inferencing terpisah yang melakukan `import training_pipeline`.

Dengan menjadikan `training_pipeline.py` murni sebagai modul yang di-*import*
(bukan dieksekusi langsung), seluruh kelas di dalamnya konsisten terikat ke
modul `training_pipeline`, sehingga artefak `.pkl` dapat dipakai ulang dengan
aman oleh skrip inferencing mana pun yang melakukan `import training_pipeline`.

Cara pakai:
    python run_training.py --data-path data_D.csv \
        --mlflow-tracking-uri sqlite:///mlflow.db \
        --experiment-name credit_score_experiment
================================================================================
"""

from b_TrainingPipeline import main

if __name__ == "__main__":
    main()
