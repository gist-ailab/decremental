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
  
    self.log = ""

  def logging(self, name, value):
    if self.is_neptune:
      self.nlogger[name].log(value)

    self.log += "{}: {:.6f}\n".format(name, value)

  def __str__(self):
    log = self.log
    self.log = ""
    return log

