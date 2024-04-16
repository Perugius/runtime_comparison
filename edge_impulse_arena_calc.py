import edgeimpulse as ei

ei.API_KEY = "ei_b8eb0c7b47b3d1c0140bdd82516d53f3f500fee0b07f5c2bdcf6b4ca7f3afd29"

print(ei.model.list_profile_devices())
print(ei.model.list_deployment_targets())
# ei.model.deploy(model="models/model_13_egc.tflite",
#                 model_input_type=ei.model.input_type.OtherInput(),
#                 model_output_type=ei.model.output_type.Regression(),
#                 output_directory='.',
#                 deploy_target='raspberry-pi-rp2040')
profile = ei.model.profile(model="models\DFKI_model_V5.tflite", device='arduino-nano-33-ble')
print(profile.summary())
