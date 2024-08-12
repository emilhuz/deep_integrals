# deep-integrals

This library can be used to generate datasets of indefinite integrals of arbitrary length and finetune the Llama-2 transformer model on them using the LORA fine tuning technique.

Model parameters belonging to the 7B version are available on https://huggingface.co/emilhuzjak/llama-2-7b-integrals.

## Dataset

Make a dataset of functions and their antiderivatives by running
```
python dataset_utils/function_generator.py --file_path <filename> --num_examples <num>
```

Or play with existing examples from the Huggingface repository.

## Model

Train the model on CUDA hardware with the following command:
```
python trainer.py --model_src metallama/
Llama-2-7b-chat-hf --checkpoint_folder emilhuzjak/llama-2-7b-integrals --save_dir <new-path> --tokenizer_src <tok-path> --train_dataset <dataset-path>
```

Alternatively, try out a larger model and leave out the ```checkpoint_folder``` to make your own fine-tuned weights.