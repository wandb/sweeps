from .. import stop_runs, next_run, RunState, SweepRun


def test_hyperband_min_iter_bands():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
            "eta": 3,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"loss": 10} for _ in range(4)]
    run2 = next_run(sweep_config, [run])
    run2.state = RunState.running
    run2.history = [{"loss": 10 - i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run, run2])

    assert to_stop[0] is run
    assert to_stop[0].early_terminate_info["bands"][:3] == [3, 9, 27]


def test_hyperband_min_iter_bands_max():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "early_terminate": {
            "type": "hyperband",
            "min_iter": 3,
            "eta": 3,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"accuracy": 10} for _ in range(4)]
    run2 = next_run(sweep_config, [run])
    run2.state = RunState.running
    run2.history = [{"accuracy": 10 + i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run, run2])

    assert to_stop[0] is run
    assert to_stop[0].early_terminate_info["bands"][:3] == [3, 9, 27]


def test_hyperband_max_iter_bands():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 81,
            "eta": 3,
            "s": 3,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"loss": 10} for _ in range(4)]
    run2 = next_run(sweep_config, [run])
    run2.state = RunState.running
    run2.history = [{"loss": 10 - i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run, run2])

    assert to_stop[0] is run
    assert to_stop[0].early_terminate_info["bands"][:3] == [3, 9, 27]


def test_init_from_max_iter():
    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 18,
            "eta": 3,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"loss": 10} for _ in range(4)]
    run2 = next_run(sweep_config, [run])
    run2.state = RunState.running
    run2.history = [{"loss": 10 - i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run, run2])

    assert to_stop[0] is run
    assert to_stop[0].early_terminate_info["bands"] == [2, 6]


def test_single_run():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 18,
            "eta": 3,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"loss": 10 - i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run])
    assert len(to_stop) == 0


def test_2runs_band1_pass():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 18,
            "eta": 3,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    run = next_run(sweep_config, [])
    run.state = RunState.running
    run.history = [{"loss": 10}, {"loss": 10}, {"loss": 6}]
    run2 = next_run(sweep_config, [run])
    run2.state = RunState.running
    run2.history = [{"loss": 10 - i} for i in range(10)]
    to_stop = stop_runs(sweep_config, [run, run2])
    assert len(to_stop) == 0


def test_5runs_band1_stop_2():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 5,
            "eta": 2,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    runs = [
        SweepRun(
            name="a",
            state=RunState.finished,  # This wont be stopped because already stopped
            history=[
                {"loss": 10},
                {"loss": 9},
            ],
        ),
        SweepRun(
            name="b",
            state=RunState.running,  # This should be stopped
            history=[
                {"loss": 10},
                {"loss": 10},
            ],
        ),
        SweepRun(
            name="c",
            state=RunState.running,  # This passes band 1 but not band 2
            history=[
                {"loss": 10},
                {"loss": 8},
                {"loss": 8},
            ],
        ),
        SweepRun(
            name="d",
            state=RunState.running,
            history=[
                {"loss": 10},
                {"loss": 7},
                {"loss": 7},
            ],
        ),
        SweepRun(
            name="e",
            state=RunState.finished,
            history=[
                {"loss": 10},
                {"loss": 6},
                {"loss": 6},
            ],
        ),
    ]

    to_stop = stop_runs(sweep_config, runs)
    assert to_stop == runs[1:3]


def test_5runs_band1_stop_2_1stnoband():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 5,
            "eta": 2,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    runs = [
        SweepRun(
            name="a",
            state=RunState.finished,  # This wont be stopped because already stopped
            history=[
                {"loss": 10},
            ],
        ),
        SweepRun(
            name="b",
            state=RunState.running,  # This should be stopped
            history=[
                {"loss": 10},
                {"loss": 10},
            ],
        ),
        SweepRun(
            name="c",
            state=RunState.running,  # This passes band 1 but not band 2
            history=[
                {"loss": 10},
                {"loss": 8},
                {"loss": 8},
            ],
        ),
        SweepRun(
            name="d",
            state=RunState.running,
            history=[
                {"loss": 10},
                {"loss": 7},
                {"loss": 7},
            ],
        ),
        SweepRun(
            name="e",
            state=RunState.finished,
            history=[
                {"loss": 10},
                {"loss": 6},
                {"loss": 6},
            ],
        ),
    ]

    to_stop = stop_runs(sweep_config, runs)
    assert to_stop == runs[1:3]


def test_eta_3():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "minimize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 9,
            "eta": 3,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    runs = [
        SweepRun(
            name="a",
            state=RunState.finished,  # This wont be stopped because already stopped
            history=[
                {"loss": 10},
                {"loss": 9},
            ],
        ),
        SweepRun(
            name="b",
            state=RunState.running,  # This should be stopped
            history=[
                {"loss": 10},
                {"loss": 10},
            ],
        ),
        SweepRun(
            name="c",
            state=RunState.running,  # This fails the first threeshold but snuck in so we wont kill
            history=[
                {"loss": 10},
                {"loss": 8},
                {"loss": 8},
                {"loss": 3},
            ],
        ),
        SweepRun(
            name="d",
            state=RunState.running,
            history=[
                {"loss": 10},
                {"loss": 7},
                {"loss": 7},
                {"loss": 4},
            ],
        ),
        SweepRun(
            name="e",
            state=RunState.running,  # this passes band 1 but doesn't pass band 2
            history=[
                {"loss": 10},
                {"loss": 6},
                {"loss": 6},
                {"loss": 6},
            ],
        ),
    ]

    # bands are at 1 and 3, thresholds are 7 and 4
    to_stop = stop_runs(sweep_config, runs)
    assert to_stop == [runs[1], runs[-1]]


def test_eta_3_max():

    sweep_config = {
        "method": "grid",
        "metric": {"name": "loss", "goal": "maximize"},
        "early_terminate": {
            "type": "hyperband",
            "max_iter": 9,
            "eta": 3,
            "s": 2,
        },
        "parameters": {"a": {"values": [1, 2, 3]}},
    }

    runs = [
        SweepRun(
            name="a",
            state=RunState.finished,  # This wont be stopped because already stopped
            history=[
                {"loss": -10},
                {"loss": -9},
            ],
        ),
        SweepRun(
            name="b",
            state=RunState.running,  # This should be stopped
            history=[
                {"loss": -10},
                {"loss": -10},
            ],
        ),
        SweepRun(
            name="c",
            state=RunState.running,  # This fails the first threeshold but snuck in so we wont kill
            history=[
                {"loss": -10},
                {"loss": -8},
                {"loss": -8},
                {"loss": -3},
            ],
        ),
        SweepRun(
            name="d",
            state=RunState.running,
            history=[
                {"loss": -10},
                {"loss": -7},
                {"loss": -7},
                {"loss": -4},
            ],
        ),
        SweepRun(
            name="e",
            state=RunState.running,  # this passes band 1 but doesn't pass band 2
            history=[
                {"loss": -10},
                {"loss": -6},
                {"loss": -6},
                {"loss": -6},
            ],
        ),
    ]

    # bands are at 1 and 3, thresholds are 7 and 4
    to_stop = stop_runs(sweep_config, runs)
    assert to_stop == [runs[1], runs[-1]]


def test_hyperband_runs_with_nan_metrics():
    # fixes https://sentry.io/share/issue/e6e002283c0447d6ac4defaf58a5d665/
    runs = [
        SweepRun(
            name="c",
            state=RunState.running,
            history=[
                {"overall_eval_loss": 19.68895149230957},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
                {"overall_eval_loss": "NaN"},
            ],
        ),
        SweepRun(
            name="d",
            state=RunState.finished,
            history=[
                {"overall_eval_loss": 6.214405059814453},
                {"overall_eval_loss": 6.050266265869141},
                {"overall_eval_loss": 5.525775909423828},
                {"overall_eval_loss": 5.143455982208252},
                {"overall_eval_loss": 4.64034366607666},
                {"overall_eval_loss": 4.741260528564453},
                {"overall_eval_loss": 4.611443519592285},
                {"overall_eval_loss": 4.968087673187256},
                {"overall_eval_loss": 4.743710517883301},
                {"overall_eval_loss": 4.415197849273682},
                {"overall_eval_loss": 4.741386413574219},
                {"overall_eval_loss": 5.067233085632324},
                {"overall_eval_loss": 4.5482563972473145},
                {"overall_eval_loss": 4.0536723136901855},
                {"overall_eval_loss": 4.09683895111084},
                {"overall_eval_loss": 4.25504732131958},
                {"overall_eval_loss": 4.438877582550049},
                {"overall_eval_loss": 4.651344299316406},
                {"overall_eval_loss": 3.8646578788757324},
                {"overall_eval_loss": 4.26999568939209},
                {"overall_eval_loss": 4.551799774169922},
                {"overall_eval_loss": 4.867758274078369},
                {"overall_eval_loss": 4.54986572265625},
                {"overall_eval_loss": 4.639401912689209},
                {"overall_eval_loss": 4.570009231567383},
                {"overall_eval_loss": 4.1927008628845215},
                {"overall_eval_loss": 4.218729496002197},
                {"overall_eval_loss": 4.279715538024902},
                {"overall_eval_loss": 4.14383602142334},
                {"overall_eval_loss": 4.373849391937256},
                {"overall_eval_loss": 4.477828025817871},
                {"overall_eval_loss": 4.309680461883545},
                {"overall_eval_loss": 4.336307048797607},
                {"overall_eval_loss": 4.2981953620910645},
                {"overall_eval_loss": 4.259463310241699},
                {"overall_eval_loss": 4.340827465057373},
                {"overall_eval_loss": 4.13071870803833},
                {"overall_eval_loss": 4.145589351654053},
                {"overall_eval_loss": 4.247255802154541},
                {"overall_eval_loss": 4.490720272064209},
                {"overall_eval_loss": 4.3509697914123535},
                {"overall_eval_loss": 4.2368855476379395},
                {"overall_eval_loss": 3.785416841506958},
                {"overall_eval_loss": 4.1248555183410645},
                {"overall_eval_loss": 4.16594123840332},
                {"overall_eval_loss": 4.210123062133789},
                {"overall_eval_loss": 4.134521007537842},
                {"overall_eval_loss": 4.481637954711914},
                {"overall_eval_loss": 3.8986079692840576},
                {"overall_eval_loss": 4.7607035636901855},
                {"overall_eval_loss": 4.429342746734619},
                {"overall_eval_loss": 4.2787957191467285},
                {"overall_eval_loss": 4.423038482666016},
                {"overall_eval_loss": 4.479197978973389},
                {"overall_eval_loss": 4.2208333015441895},
                {"overall_eval_loss": 4.325165748596191},
                {"overall_eval_loss": 4.398067951202393},
                {"overall_eval_loss": 4.34964656829834},
                {"overall_eval_loss": 4.7197394371032715},
                {"overall_eval_loss": 4.056498050689697},
                {"overall_eval_loss": 4.42212438583374},
            ],
        ),
        SweepRun(
            name="e",
            state=RunState.finished,  # this passes band 1 but doesn't pass band 2
            history=[
                {"overall_eval_loss": 6.784717559814453},
                {"overall_eval_loss": 5.467315196990967},
                {"overall_eval_loss": 4.862574577331543},
                {"overall_eval_loss": 5.149237155914307},
                {"overall_eval_loss": 4.713217258453369},
                {"overall_eval_loss": 4.645257472991943},
                {"overall_eval_loss": 4.654730319976807},
                {"overall_eval_loss": 4.218390464782715},
                {"overall_eval_loss": 4.540946960449219},
                {"overall_eval_loss": 4.710428714752197},
                {"overall_eval_loss": 4.327051162719727},
                {"overall_eval_loss": 4.395434379577637},
                {"overall_eval_loss": 4.212771892547607},
                {"overall_eval_loss": 4.131608963012695},
                {"overall_eval_loss": 4.5795159339904785},
                {"overall_eval_loss": 4.4573750495910645},
                {"overall_eval_loss": 4.383418560028076},
                {"overall_eval_loss": 4.822948455810547},
                {"overall_eval_loss": 4.4155707359313965},
                {"overall_eval_loss": 4.176084995269775},
                {"overall_eval_loss": 4.340216636657715},
                {"overall_eval_loss": 4.1993794441223145},
                {"overall_eval_loss": 4.623438835144043},
                {"overall_eval_loss": 4.61974573135376},
                {"overall_eval_loss": 4.306433200836182},
                {"overall_eval_loss": 4.174667835235596},
                {"overall_eval_loss": 4.4847493171691895},
                {"overall_eval_loss": 3.8551318645477295},
                {"overall_eval_loss": 4.469892978668213},
                {"overall_eval_loss": 4.2765727043151855},
                {"overall_eval_loss": 4.316288948059082},
                {"overall_eval_loss": 4.273141384124756},
                {"overall_eval_loss": 4.58651876449585},
                {"overall_eval_loss": 4.162075519561768},
                {"overall_eval_loss": 4.0801591873168945},
                {"overall_eval_loss": 4.499518871307373},
                {"overall_eval_loss": 4.574384689331055},
                {"overall_eval_loss": 4.025749683380127},
                {"overall_eval_loss": 4.137139320373535},
                {"overall_eval_loss": 4.478872299194336},
                {"overall_eval_loss": 4.130046367645264},
                {"overall_eval_loss": 4.551371097564697},
                {"overall_eval_loss": 4.507683753967285},
                {"overall_eval_loss": 4.316148281097412},
                {"overall_eval_loss": 4.449378490447998},
                {"overall_eval_loss": 4.43292236328125},
                {"overall_eval_loss": 3.856088876724243},
                {"overall_eval_loss": 3.8799805641174316},
                {"overall_eval_loss": 4.233527660369873},
                {"overall_eval_loss": 4.096866607666016},
                {"overall_eval_loss": 4.5571112632751465},
                {"overall_eval_loss": 4.188418865203857},
                {"overall_eval_loss": 3.938608407974243},
                {"overall_eval_loss": 4.292693138122559},
                {"overall_eval_loss": 4.055037021636963},
                {"overall_eval_loss": 4.307272434234619},
                {"overall_eval_loss": 4.0329108238220215},
                {"overall_eval_loss": 4.152763366699219},
                {"overall_eval_loss": 4.2551350593566895},
                {"overall_eval_loss": 3.7090814113616943},
                {"overall_eval_loss": 4.495356559753418},
            ],
        ),
    ]
    config = {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "overall_eval_loss"},
        "parameters": {
            "lr": {"values": [0.001, 0.0003, 0.0001, 3e-05]},
            "pred_out_channels": {"max": 64, "min": 1, "distribution": "int_uniform"},
            "ia_char_embedding_size": {
                "max": 16,
                "min": 1,
                "distribution": "int_uniform",
            },
            "ja_char_embedding_size": {
                "max": 16,
                "min": 1,
                "distribution": "int_uniform",
            },
            "pred_resnet_type_index": {"values": [0, 1, 2]},
            "sr_pred_hidden_channels": {
                "max": 64,
                "min": 1,
                "distribution": "int_uniform",
            },
            "ia_mental_embedding_size": {
                "max": 64,
                "min": 1,
                "distribution": "int_uniform",
            },
            "ja_mental_embedding_size": {
                "max": 64,
                "min": 1,
                "distribution": "int_uniform",
            },
            "goal_pred_hidden_channels": {
                "max": 64,
                "min": 1,
                "distribution": "int_uniform",
            },
            "ia_char_hidden_input_size": {
                "max": 64,
                "min": 1,
                "distribution": "int_uniform",
            },
            "ia_char_resnet_type_index": {"values": [0, 1, 2]},
            "ja_char_hidden_input_size": {
                "max": 64,
                "min": 1,
                "distribution": "int_uniform",
            },
            "ja_char_resnet_type_index": {"values": [0, 1, 2]},
            "ia_char_hidden_output_size": {
                "max": 128,
                "min": 1,
                "distribution": "int_uniform",
            },
            "ja_char_hidden_output_size": {
                "max": 128,
                "min": 1,
                "distribution": "int_uniform",
            },
            "action_pred_hidden_channels": {
                "max": 64,
                "min": 1,
                "distribution": "int_uniform",
            },
            "ia_mental_hidden_input_size": {
                "max": 64,
                "min": 1,
                "distribution": "int_uniform",
            },
            "ia_mental_resnet_type_index": {"values": [0, 1, 2]},
            "ja_mental_hidden_input_size": {
                "max": 64,
                "min": 1,
                "distribution": "int_uniform",
            },
            "ja_mental_resnet_type_index": {"values": [0, 1, 2]},
            "ia_mental_hidden_output_size": {
                "max": 64,
                "min": 1,
                "distribution": "int_uniform",
            },
            "ja_mental_hidden_output_size": {
                "max": 64,
                "min": 1,
                "distribution": "int_uniform",
            },
            "pred_hidden_channel_type_index": {
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            },
            "ia_char_hidden_channel_type_index": {
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            },
            "ja_char_hidden_channel_type_index": {
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            },
            "ia_mental_hidden_channel_type_index": {
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            },
            "ja_mental_hidden_channel_type_index": {
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            },
        },
        "early_terminate": {"s": 2, "eta": 3, "type": "hyperband", "max_iter": 300},
    }
    assert stop_runs(config, runs) == []
