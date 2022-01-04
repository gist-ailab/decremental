import neptune.new as neptune


class Logger:
  def __init__(self, cfg, is_neptune):
    self.is_neptune = is_neptune
    if is_neptune:
      self.nlogger = neptune.init(
        project=cfg["project"],
        api_token=cfg["token"]
      )
      self.nlogger["parameters"] = cfg
    else:
      self.nlogger = None
  
  def logging(self, name, value):
    if self.is_neptune:
      self.nlogger[name].log(value)
  