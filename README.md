# SIH-DVS-HAR

Human activity recognition on DVS data

## Todo

--> Drive connectivity

--> Actual deployment

--> Preprocess data:

	--> Convert data to frames

	--> Concatenate into 3D Tensors

--> Train model

--> Connect all components

## Usage

Structure data as shown in the below given example.

```bash
data/
.
└── train/
	└── label1/
		└── sample1.npy
		└── sample2.npy
		└── ...
	└── label2/
		└── sample1.npy
		└── sample2.npy
		└── ...
	└── ...
└── test/
	└── label1/
		└── sample1.npy
		└── sample2.npy
		└── ...
	└── label2/
		└── sample1.npy
		└── sample2.npy
		└── ...		
	└── ...
```