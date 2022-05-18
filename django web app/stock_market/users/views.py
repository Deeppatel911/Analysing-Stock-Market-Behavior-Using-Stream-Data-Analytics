from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from . forms import UserRegisterForm, UserUpdateForm
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login

# Create your views here.


def login_user(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user:
            login(request, user)
            return redirect('market_home')

        else:
            messages.error(request, 'Invalid credentials!')
            return redirect('login')

    else:
        return render(request, 'users/login.html')


def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        fname = request.POST['first_name']
        lname = request.POST['last_name']
        email = request.POST['email']
        pass1 = request.POST['password1']
        pass2 = request.POST['password2']

        if pass2 == pass1:
            if User.objects.filter(username=username).exists():
                messages.error(request, 'User is already Registered!')
                return redirect('register')

            else:
                new_user = User.objects.create_user(username=username, password=pass1, first_name=fname,
                                                    last_name=lname, email=email)
                new_user.save()
                messages.success(request, "Your account has been created! You're now able to login.")
                return redirect('login')

        else:
            messages.error(request, 'Passwords do not match!')
            return redirect('register')

    else:
        return render(request, 'users/register.html')

# def register(request):
#     if request.method == 'POST':
#         form = UserRegisterForm(request.POST)
#
#         if form.is_valid():
#             form.save()
#             messages.success(request, "Your account has been created! You're now able to login.")
#             return redirect('login')
#     else:
#         form = UserRegisterForm()
#     return render(request, 'users/register.html', {'form': form})


@login_required
def profile(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, instance=request.user)

        if u_form.is_valid():
            u_form.save()
            messages.success(request, "Your account has been updated!")
            return redirect('profile')

    else:
        u_form = UserUpdateForm(instance=request.user)
    return render(request, 'users/profile.html', {'u_form': u_form})

