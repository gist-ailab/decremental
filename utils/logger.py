import neptune.new as neptune


class Logger:
  def __init__(self, cfg, neptune):
    self.neptune = neptune
    if neptune:
      self.nlogger = neptune.init(
        project=cfg["project"],
        api_token=cfg["token"]
      )
      self.nlogger["parameters"] = cfg
    else:
      self.nlogger = None
  