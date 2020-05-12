from virtualmouse.constants import ModelConsts

def SaveModel(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(ModelConsts.MODEL_JS_LOCATION, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(ModelConsts.MODEL_WEIGHTS_LOCATION)


def CompileModel(model):
    import keras
    model.compile(optimizer="adam" , loss= keras.losses.mse)
    return model


def LoadModel():
    from keras.models import model_from_json

    json_file = open(ModelConsts.MODEL_JS_LOCATION, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights(ModelConsts.MODEL_WEIGHTS_LOCATION)
    print("Loaded model from disk")

    return CompileModel(loaded_model)
