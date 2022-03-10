class SchedDomain():
    def __init__(self, span):
        self.span = span
        self.children = []
        self.cpus = []
        self.parent = None

    def add_child(self, child):
        self.children.append(child)

    def add_cpu(self, cpu):
        self.cpus.append(cpu)

    def set_parent(self, parent):
        if self.parent:
            if self.parent != parent:
                raise Exception('Parent conflict')
        else:
            self.parent = parent

    def __repr__(self):
        #  return "({} {})".format(self.span, self.parent)
        #  return "({}, {})".format(self.children.span, self.parent.span if self.parent else None)
        return self.span


if __name__ == '__main__':
    with open('/proc/schedstat', 'r') as f:
        lines = f.readlines()[2:]

    curr = []
    cpus = {}
    cpu = None
    for line in lines:
        if line[:3] == 'cpu':
            if cpu != None:
                cpus[cpu] = curr
                curr = []
            cpu = int(line.split(' ')[0][3:])
        elif line[:6] == 'domain':
            curr.append(line.split(' ')[1])
        else:
            print('This should not happen')
    if cpu:
        cpus[cpu] = curr

    domains = {}
    for cpu, spans in cpus.items():
        last = None
        for span in spans:
            if span in domains:
                domain = domains[span]
                domain.add_cpu(cpu)
            else:
                domain = SchedDomain(span)
                domains[span] = domain
                domain.add_cpu(cpu)
            if last:
                domain.add_child(last)
                last.set_parent(domain)
            last = domain

    for span, domain in domains.items():
        print(span, domain.cpus, [child.span for child in domain.children], domain.parent)
