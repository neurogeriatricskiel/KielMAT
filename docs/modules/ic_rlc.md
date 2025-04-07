KielMAT provides two alternative methods for classifying the laterality of initial contacts (i.e., distinguishing left and right foot contacts) based on gyroscope signals recorded from a lower-back IMU sensor. Both methods operate on initial contact timestamps previously detected using a separate algorithm (e.g., Paraschiv-Ionescu).

- The **McCamley method** is a rule-based approach that uses the sign of the angular velocity signal at the time of contact to infer left or right contacts.

- The **Ullrich method** is a data-driven approach that uses a machine learning model trained on filtered gyroscope signals and their derivatives.

The two methods differ mainly in complexity and required resources. Users can choose the method best suited to their dataset and analysis goals.

## Initial Contact Classification (McCamley)

::: modules.rlc._mccamley.MacCamleyInitialContactClassification

## Initial Contact Classification (Ulrich)

::: modules.rlc._ulrich.UllrichInitialContactClassification