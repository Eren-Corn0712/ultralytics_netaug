import torch
import torch.nn as nn


def yolo_export(netaug_model, width_scaler: list):
    if hasattr(netaug_model, "set_active"):
        netaug_model.out_ch = netaug_model.set_active(width_scaler)
        export_model = []
        for m in netaug_model.model:
            if "Dynamic" in m.__class__.__name__:
                m_export = m.export_module()
                for a, v in m.__dict__.items():
                    if a in ("f", "i", "np", "type"):
                        setattr(m_export, a, v)
                export_model.append(m_export)
            else:
                export_model.append(m)
        export_model = nn.Sequential(*export_model)
        return export_model
    else:
        raise ValueError(f"The model haven't set active method!")
